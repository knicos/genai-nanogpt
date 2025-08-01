import { ITokeniser } from './Tokeniser/type';
import { DatasetBuilder } from './DatasetBuilder';
import NanoGPT, { TrainingLogEntry } from './NanoGPTModel';
import type TF from '@tensorflow/tfjs';

export interface TrainingState {
    epoch: number;
    step: number;
    lastLoss: number;
    epochLoss: number;
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
    epochs: number;
    stepsPerEpoch: number;
    desiredLoss: number;
    logInterval: number;
    prompt?: string;
    onEpoch?: (e: number, loss: number, valLoss?: number) => Promise<void> | void;
    onStep?: (log: TrainingLogEntry) => Promise<void> | void;
}

// Enhanced training utilities with Dataset API and memory leak fixes
export default abstract class GPTTrainer {
    protected model: NanoGPT;
    protected optimizer: TF.Optimizer;
    protected datasetBuilder: DatasetBuilder;
    protected tf: typeof TF;
    protected learningRate: number;

    constructor(tf: typeof TF, model: NanoGPT, tokenizer: ITokeniser, learningRate: number = 3e-4) {
        this.tf = tf;
        this.model = model;
        this.learningRate = learningRate;
        this.optimizer = this.tf.train.adam(learningRate, 0.9, 0.999, 1e-8);
        this.datasetBuilder = new DatasetBuilder(this.tf, tokenizer, model.config.blockSize);
    }

    resetOptimizer(config: AdamConfig = { learningRateFactor: 1, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }): void {
        this.optimizer.dispose();
        this.optimizer = this.tf.train.adam(
            config.learningRateFactor * this.learningRate,
            config.beta1,
            config.beta2,
            config.epsilon
        );
    }

    protected trainStep(batch: { xs: TF.Tensor; ys: TF.Tensor }, dummy = false): TF.Scalar {
        return this.tf.tidy(() => {
            const { xs, ys } = batch;

            const f = () => {
                const { loss } = this.model.forward(xs, ys, true);
                return loss! as TF.Scalar;
            };

            const { value: lossValue, grads } = this.optimizer.computeGradients(f);

            if (!dummy) {
                // Apply gradients
                this.optimizer.applyGradients(grads);
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
            l.dispose(); // Dispose loss to free memory
        } catch (error) {
            console.error('Error during dummy pass:', error);
        } finally {
            dummyBatch.dispose();
            dummyTargets.dispose(); // Dispose dummy targets to free memory
        }
    }

    protected trainBatch(state: TrainingState, batch: { xs: TF.Tensor; ys: TF.Tensor }): number {
        try {
            const lossScalar = this.trainStep(batch);
            const data = lossScalar.arraySync();
            lossScalar.dispose();
            batch.xs.dispose();
            batch.ys.dispose();
            state.epochLoss += data;
            state.lastLoss = data;
            state.losses.push(data);
            state.step++;
            state.totalSteps++;
            return data;
        } catch (error) {
            console.error(`Error processing batch at step ${state.step}:`, error);
            this.tf.dispose();
            throw error;
        }
    }

    // Train for multiple epochs using Dataset API - FIXED memory leaks
    abstract trainOnDataset(
        dataset: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>,
        options: Partial<TrainingOptions>,
        validationDataset?: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>
    ): Promise<{ losses: number[]; validationLosses: number[] }>;

    // Evaluate model on validation dataset - FIXED memory leaks
    async evaluateOnDataset(dataset: TF.data.Dataset<TF.TensorContainer>, maxBatches: number = 100): Promise<number> {
        let totalLoss = 0;
        let batchCount = 0;

        await dataset.take(maxBatches).forEachAsync(async (batch) => {
            const { xs, ys } = batch as { xs: TF.Tensor; ys: TF.Tensor };

            // FIX 8: Manual tensor disposal for async operations
            const { loss, logits } = this.model.forward(xs, ys, false);
            const lossValue = loss!.arraySync();
            const batchLoss = lossValue as number;

            // FIX 9: Dispose loss tensor immediately after use
            loss!.dispose();
            logits.dispose(); // Dispose logits if not needed

            totalLoss += batchLoss;
            batchCount++;

            // Batch tensors are disposed by the dataset iterator
        });

        return totalLoss / batchCount;
    }

    // Create training and validation datasets - FIXED memory leaks
    async createTrainValidationSplit(
        textData: string[],
        batchSize: number = 32,
        validationSplit: number = 0.1
    ): Promise<{
        trainDataset: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>;
        validationDataset: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>;
    }> {
        const splitIndex = Math.floor(textData.length * (1 - validationSplit));

        const trainTexts = textData.slice(0, splitIndex);
        const validationTexts = textData.slice(splitIndex);

        const trainDataset = await this.datasetBuilder.createTextDataset(trainTexts, batchSize);
        const validationDataset = await this.datasetBuilder.createTextDataset(validationTexts, batchSize);

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
        // Force cleanup of any remaining tensors
        this.tf.dispose();
    }
}
