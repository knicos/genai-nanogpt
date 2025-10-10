import type { ITokeniser } from '../tokeniser/type';
import { DatasetBuilder, flattenTokens, PAGE_FACTOR } from './DatasetBuilder';
import NanoGPT, { TrainingLogEntry } from '../NanoGPTModel';
import AdamExt from './AdamExt';
import { NamedVariableMap, TensorContainer } from '@tensorflow/tfjs-core/dist/tensor_types';
import { dispose, norm, Scalar, Tensor, tidy, variableGrads, zeros } from '@tensorflow/tfjs-core';
import { Dataset } from '@tensorflow/tfjs-data';

export interface TrainingState {
    step: number;
    lastLoss: number;
    totalSteps: number;
    losses: number[];
    validationLosses: number[];
    gradientNorm?: number;
}

export interface TrainingProgress {
    duration: number;
    totalSamples: number;
    samplesPerSecond: number;
    memory?: number;
}

export interface AdamConfig {
    learningRateFactor: number;
    beta1: number;
    beta2: number;
    epsilon: number;
}

export interface TrainingOptions {
    desiredLoss: number;
    logInterval: number;
    prompt?: string;
    maxSteps: number;
    advancedMetrics?: boolean;
    onStep?: (log: TrainingLogEntry, progress: TrainingProgress) => Promise<void> | void;
}

// Enhanced training utilities with Dataset API and memory leak fixes
export default abstract class GPTTrainer {
    protected model: NanoGPT;
    protected optimizer!: AdamExt;
    protected datasetBuilder: DatasetBuilder;
    protected learningRate: number;
    protected running = false;
    protected lastState?: TrainingState;

    constructor(model: NanoGPT, protected tokenizer: ITokeniser, learningRate: number = 1e-3) {
        this.model = model;
        this.learningRate = learningRate;
        this.resetOptimizer();
        this.datasetBuilder = new DatasetBuilder(tokenizer, model.config.gpt.blockSize);
    }

    setLearningRate(learningRate: number): void {
        this.learningRate = learningRate;
        this.resetOptimizer({ learningRateFactor: 1, beta1: 0.9, beta2: 0.99, epsilon: 1e-8 });
    }

    reset() {
        this.lastState = undefined;
        this.running = false;
    }

    stop() {
        this.running = false;
    }

    getOptimizer(): AdamExt {
        return this.optimizer;
    }

    resetOptimizer(config: AdamConfig = { learningRateFactor: 1, beta1: 0.9, beta2: 0.99, epsilon: 1e-8 }): void {
        if (this.optimizer) this.optimizer.dispose();
        const adam = new AdamExt(
            config.learningRateFactor * this.learningRate,
            config.beta1,
            config.beta2,
            config.epsilon,
            {
                warmupSteps: 100,
                decaySteps: 20000,
                minLearningRate: 1e-4,
                weightDecay: 0,
            }
        );
        this.optimizer = adam;
    }

    private maxGradNorm(grads: NamedVariableMap): number {
        let maxNorm = 0;
        // Print all gradients
        Object.keys(grads).forEach((varName) => {
            const grad = grads[varName];
            const temp = norm(grad);
            const gradNorm = temp.dataSync()[0];
            temp.dispose();
            if (gradNorm > maxNorm) {
                maxNorm = gradNorm;
            }
        });
        return maxNorm;
    }

    protected trainStep(
        state: Partial<TrainingState>,
        batch: { xs: Tensor; ys: Tensor },
        dummy = false,
        calcNorm = false
    ): Scalar {
        return tidy(() => {
            this.model.getProfiler()?.startMemory();
            const { xs, ys } = batch;

            const f = () => {
                const [logits, loss] = this.model.forward({ training: true }, xs, ys);
                //console.log('Logits', logits.toString());
                logits.dispose();
                return loss! as Scalar;
            };

            //const vars = this.model.variables;
            const { value: lossValue, grads } = variableGrads(f);

            if (!dummy) {
                // Clip gradients
                /*const clippedGrads: { [variableName: string]: TF.Tensor } = {};
                for (const variableName in grads) {
                    clippedGrads[variableName] = this.tf.clipByValue(grads[variableName], -1.0, 1.0);
                }*/

                if (calcNorm) {
                    const maxNorm = this.maxGradNorm(grads as NamedVariableMap);
                    state.gradientNorm = maxNorm;
                }
                //this.tf.dispose(grads);
                // Apply gradients
                this.optimizer.applyGradients(grads as NamedVariableMap);

                this.model.getProfiler()?.endMemory('Training');

                dispose(grads);
            } else {
                this.model.getProfiler()?.endMemory('Training');
            }

            return lossValue;
        });
    }

    protected dummyPass(): void {
        // Send a dummy input to initialize the model
        const dummyBatch = zeros([1, this.model.config.gpt.blockSize], 'int32');
        const dummyTargets = zeros([1, this.model.config.gpt.blockSize], 'int32');

        try {
            const l = this.trainStep({}, { xs: dummyBatch, ys: dummyTargets }, true);
            l.dataSync(); // Ensure loss is computed
            l.dispose(); // Dispose loss to free memory
        } catch (error) {
            console.error('Error during dummy pass:', error);
        } finally {
            dummyBatch.dispose();
            dummyTargets.dispose(); // Dispose dummy targets to free memory
        }
    }

    protected async trainBatch(
        state: TrainingState,
        batch: { xs: Tensor; ys: Tensor },
        calcNorm = false
    ): Promise<number> {
        try {
            //console.log('Batch XS', batch.xs.toString());
            const lossScalar = this.trainStep(state, batch, false, calcNorm);
            batch.xs.dispose();
            batch.ys.dispose();

            state.step++;
            state.totalSteps++;

            return lossScalar.array().then((lossValue) => {
                state.lastLoss = lossValue as number;
                state.losses.push(state.lastLoss);
                lossScalar.dispose();

                return state.lastLoss;
            });
        } catch (error) {
            console.error(`Error processing batch at step ${state.step}:`, error);
            dispose();
            throw error;
        }
    }

    abstract trainOnDataset(
        dataset: Dataset<{ xs: Tensor; ys: Tensor }>,
        options: Partial<TrainingOptions>,
        validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>
    ): Promise<{ losses: number[]; validationLosses: number[] }>;

    async createTrainValidationSplit(
        textData: string[],
        batchSize: number = 32,
        validationSplit: number = 0.1
    ): Promise<{
        trainDataset: Dataset<{ xs: Tensor; ys: Tensor }>;
        validationDataset: Dataset<{ xs: Tensor; ys: Tensor }>;
    }> {
        const allTokens = await flattenTokens(textData, this.tokenizer);

        const validationMask = new Set<number>();
        if (validationSplit > 0) {
            const totalPages = Math.floor(allTokens.length / (this.datasetBuilder.blockSize * PAGE_FACTOR));
            const numValidationPages = Math.max(1, Math.floor(totalPages * validationSplit));

            while (validationMask.size < numValidationPages) {
                const pageIndex = Math.floor(Math.random() * totalPages);
                validationMask.add(pageIndex);
            }
        }

        const trainDataset = await this.datasetBuilder.createTextDataset(allTokens, batchSize, validationMask, false);
        const validationDataset = await this.datasetBuilder.createTextDataset(
            allTokens,
            batchSize,
            validationMask,
            true
        );

        return { trainDataset, validationDataset };
    }

    async createDataset(textData: string[], batchSize: number = 32): Promise<Dataset<TensorContainer>> {
        const allTokens = await flattenTokens(textData, this.tokenizer);
        const trainDataset = await this.datasetBuilder.createTextDataset(allTokens, batchSize);
        return trainDataset;
    }

    dispose(): void {
        if (this.optimizer) {
            this.optimizer.dispose();
        }
    }
}
