import type { ITokeniser } from '../tokeniser/type';
import { DatasetBuilder, flattenTokens, PAGE_FACTOR } from './DatasetBuilder';
import AdamExt from './AdamExt';
import { NamedTensorMap, NamedVariableMap, TensorContainer } from '@tensorflow/tfjs-core/dist/tensor_types';
import { dispose, keep, scalar, Scalar, Tensor, tidy, variableGrads, zeros } from '@tensorflow/tfjs-core';
import { Dataset } from '@tensorflow/tfjs-data';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { TensorStatistics } from '../checks/weights';

export interface TrainingLogEntry {
    loss: number;
    valLoss?: number;
    step: number;
    time: number;
    example?: string;
    batchSize: number;
    gradientMetrics?: Map<string, TensorStatistics>;
    learningRate?: number;
}

export interface TrainingState {
    step: number;
    lastLoss: number;
    totalSteps: number;
    losses: number[];
    validationLosses: number[];
    gradients?: NamedTensorMap;
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
    gradientMetrics?: boolean;
    onStep?: (log: TrainingLogEntry, progress: TrainingProgress) => Promise<void> | void;
}

// Enhanced training utilities with Dataset API and memory leak fixes
export default abstract class GPTTrainer {
    protected model: Model<ModelForwardAttributes>;
    protected optimizer!: AdamExt;
    protected datasetBuilder: DatasetBuilder;
    protected learningRate: number;
    protected running = false;
    protected lastState?: TrainingState;
    protected _gradientCheckpointing: boolean = false;
    protected _mixedPrecision: boolean = false;
    protected lossScaling: number;

    constructor(model: Model<ModelForwardAttributes>, protected tokenizer: ITokeniser, learningRate: number = 1e-3) {
        this.model = model;
        this.lossScaling = model.lossScaling;
        this.learningRate = learningRate;
        this.resetOptimizer();
        this.datasetBuilder = new DatasetBuilder(tokenizer, model.config.blockSize);
    }

    setGradientCheckpointing(enabled: boolean): void {
        this._gradientCheckpointing = enabled;
    }

    setMixedPrecision(enabled: boolean): void {
        this._mixedPrecision = enabled;
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
                lossScaling: this.lossScaling,
            }
        );
        this.optimizer = adam;
    }

    protected trainStep(
        state: Partial<TrainingState>,
        batch: { xs: Tensor; ys: Tensor },
        dummy = false,
        keepGrads = false
    ): Scalar {
        return tidy(() => {
            this.model.getProfiler()?.startMemory();
            const { xs, ys } = batch;

            const f = () => {
                const [logits, loss] = this.model.forward(
                    {
                        training: true,
                        checkpointing: this._gradientCheckpointing,
                        mixedPrecision: this._mixedPrecision,
                    },
                    xs,
                    ys
                );
                logits.dispose();
                const scaledLoss = loss.mul(scalar(this.lossScaling));
                loss.dispose();
                return scaledLoss as Scalar;
            };

            const { value: lossValue, grads } = variableGrads(f);

            if (!dummy) {
                // Apply gradients
                this.optimizer.applyGradients(grads as NamedVariableMap);

                this.model.getProfiler()?.endMemory('Training');

                if (keepGrads) {
                    state.gradients = grads;
                    Object.values(grads).forEach((g) => keep(g));
                } else {
                    dispose(grads);
                }
            } else {
                this.model.getProfiler()?.endMemory('Training');
            }

            return lossValue.mul(scalar(1 / this.lossScaling)) as Scalar;
        });
    }

    protected async dummyPass(): Promise<void> {
        // Send a dummy input to initialize the model
        const dummyBatch = zeros([1, this.model.config.blockSize], 'int32');
        const dummyTargets = zeros([1, this.model.config.blockSize], 'int32');

        try {
            const l = this.trainStep({}, { xs: dummyBatch, ys: dummyTargets }, true);
            await l.data(); // Ensure loss is computed
            l.dispose(); // Dispose loss to free memory
        } catch (error) {
            console.error('Error during dummy pass:', error);
        } finally {
            dummyBatch.dispose();
            dummyTargets.dispose(); // Dispose dummy targets to free memory
        }
    }

    protected trainBatch(state: TrainingState, batch: { xs: Tensor; ys: Tensor }, keepGrads = false): Scalar {
        try {
            //console.log('Batch XS', batch.xs.toString());
            const lossScalar = this.trainStep(state, batch, false, keepGrads);
            batch.xs.dispose();
            batch.ys.dispose();

            state.step++;
            state.totalSteps++;

            return lossScalar;
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

    abstract stepDataset(
        dataset: Dataset<{ xs: Tensor; ys: Tensor }>,
        options: Partial<TrainingOptions>,
        validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>
    ): Promise<{ log: TrainingLogEntry; progress: TrainingProgress }>;

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
