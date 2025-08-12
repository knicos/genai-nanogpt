import { ITokeniser } from '../tokeniser/type';
import { DatasetBuilder } from './DatasetBuilder';
import NanoGPT, { TrainingLogEntry } from '../NanoGPTModel';
import type TF from '@tensorflow/tfjs';
import AdamExt from './AdamExt';
import { NamedVariableMap } from '@tensorflow/tfjs-core/dist/tensor_types';

export interface TrainingState {
    step: number;
    lastLoss: number;
    totalSteps: number;
    losses: number[];
    validationLosses: number[];
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
    onStep?: (log: TrainingLogEntry) => Promise<void> | void;
}

// Enhanced training utilities with Dataset API and memory leak fixes
export default abstract class GPTTrainer {
    protected model: NanoGPT;
    protected optimizer!: AdamExt;
    protected datasetBuilder: DatasetBuilder;
    protected tf: typeof TF;
    protected learningRate: number;
    protected running = false;
    protected lastState?: TrainingState;

    constructor(tf: typeof TF, model: NanoGPT, protected tokenizer: ITokeniser, learningRate: number = 1e-3) {
        this.tf = tf;
        this.model = model;
        this.learningRate = learningRate;
        this.resetOptimizer();
        this.datasetBuilder = new DatasetBuilder(this.tf, tokenizer, model.config.blockSize);
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

    private printGradients(grads: NamedVariableMap): void {
        // Print all gradients
        Object.keys(grads).forEach((varName) => {
            const grad = grads[varName];
            console.log(`${varName}:`);
            console.log(`  Shape: ${grad.shape}`);
            console.log(`  Mean: ${this.tf.mean(grad).dataSync()[0]}`);
            console.log(`  Std: ${this.tf.moments(grad).variance.sqrt().dataSync()[0]}`);
            console.log(`  Min: ${this.tf.min(grad).dataSync()[0]}`);
            console.log(`  Max: ${this.tf.max(grad).dataSync()[0]}`);
            console.log(`  Norm: ${this.tf.norm(grad).dataSync()[0]}`);
        });
    }

    protected trainStep(batch: { xs: TF.Tensor; ys: TF.Tensor }, dummy = false, print = false): TF.Scalar {
        return this.tf.tidy(() => {
            const { xs, ys } = batch;

            const f = () => {
                const { loss, logits } = this.model.forward(xs, ys, true);
                //console.log('Logits', logits.toString());
                logits.dispose();
                return loss! as TF.Scalar;
            };

            //const vars = this.model.variables;
            const { value: lossValue, grads } = this.tf.variableGrads(f);

            if (!dummy) {
                // Clip gradients
                /*const clippedGrads: { [variableName: string]: TF.Tensor } = {};
                for (const variableName in grads) {
                    clippedGrads[variableName] = this.tf.clipByValue(grads[variableName], -1.0, 1.0);
                }*/

                if (print) {
                    console.log('-------');
                    this.printGradients(grads as NamedVariableMap);
                    console.log('-------');
                }
                //this.tf.dispose(grads);
                // Apply gradients
                this.optimizer.applyGradients(grads as NamedVariableMap);
                this.tf.dispose(grads);
            }

            return lossValue;
        });
    }

    protected dummyPass(): void {
        // Send a dummy input to initialize the model
        const dummyBatch = this.tf.zeros([1, this.model.config.blockSize], 'int32');
        const dummyTargets = this.tf.zeros([1, this.model.config.blockSize, this.model.config.vocabSize]);

        try {
            const l = this.trainStep({ xs: dummyBatch, ys: dummyTargets }, true);
            l.dataSync(); // Ensure loss is computed
            l.dispose(); // Dispose loss to free memory
        } catch (error) {
            console.error('Error during dummy pass:', error);
        } finally {
            dummyBatch.dispose();
            dummyTargets.dispose(); // Dispose dummy targets to free memory
        }
    }

    protected async trainBatch(state: TrainingState, batch: { xs: TF.Tensor; ys: TF.Tensor }): Promise<number> {
        try {
            //console.log('Batch XS', batch.xs.toString());
            const lossScalar = this.trainStep(batch, false, false);
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
            this.tf.dispose();
            throw error;
        }
    }

    abstract trainOnDataset(
        dataset: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>,
        options: Partial<TrainingOptions>,
        validationDataset?: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>
    ): Promise<{ losses: number[]; validationLosses: number[] }>;

    async createTrainValidationSplit(
        textData: string[],
        batchSize: number = 32,
        validationSplit: number = 0.1
    ): Promise<{
        trainDataset: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>;
        validationDataset: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>;
    }> {
        const trainDataset = await this.datasetBuilder.createTextDataset(textData, batchSize, 0, 1 - validationSplit);
        const validationDataset = await this.datasetBuilder.createTextDataset(
            textData,
            batchSize,
            1 - validationSplit,
            1
        );

        return { trainDataset, validationDataset };
    }

    async createDataset(textData: string[], batchSize: number = 32): Promise<TF.data.Dataset<TF.TensorContainer>> {
        const trainDataset = await this.datasetBuilder.createTextDataset(textData, batchSize);
        return trainDataset;
    }

    dispose(): void {
        if (this.optimizer) {
            this.optimizer.dispose();
        }
    }
}
