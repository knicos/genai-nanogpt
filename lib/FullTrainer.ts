import { ITokeniser } from './Tokeniser/type';
import { generateText } from './generate';
import NanoGPT, { TrainingLogEntry } from './NanoGPTModel';
import type TF from '@tensorflow/tfjs';
import GPTTrainer, { TrainingOptions } from './Trainer';

interface TrainingState {
    epoch: number;
    pass: number;
    depth: number;
    step: number;
    stepSinceDepthChange: number;
    lastLoss: number;
    epochLoss: number;
    totalSteps: number;
    losses: number[];
    validationLosses: number[];
}

const DEFAULT_OPTIONS: TrainingOptions = {
    epochs: 1,
    stepsPerEpoch: 1000000,
    desiredLoss: 0.01,
    logInterval: 1,
};

// Enhanced training utilities with Dataset API and memory leak fixes
export default class FullTrainer extends GPTTrainer {
    constructor(tf: typeof TF, model: NanoGPT, tokenizer: ITokeniser, learningRate: number = 3e-4) {
        super(tf, model, tokenizer, learningRate);
    }

    // Train for multiple epochs using Dataset API - FIXED memory leaks
    async trainOnDataset(
        dataset: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>,
        options: Partial<TrainingOptions>,
        validationDataset?: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>
    ): Promise<{ losses: number[]; validationLosses: number[] }> {
        const { epochs, stepsPerEpoch, desiredLoss, logInterval, onStep, onEpoch, prompt } = {
            ...DEFAULT_OPTIONS,
            ...options,
        };

        const state: TrainingState = {
            epoch: 0,
            pass: 0,
            depth: 1,
            step: 0,
            stepSinceDepthChange: 0,
            lastLoss: 1e6,
            epochLoss: 0,
            totalSteps: 0,
            losses: [],
            validationLosses: [],
        };

        this.dummyPass();

        const startTime = Date.now();

        for (state.epoch = 0; state.epoch < epochs; state.epoch++) {
            state.step = 0;
            state.epochLoss = 0;
            state.pass = 0;
            state.depth = 1;
            state.stepSinceDepthChange = 0;

            const iterator = await dataset.iterator();

            // Training loop with try-catch for better error handling
            try {
                while (true) {
                    if (stepsPerEpoch && state.step >= stepsPerEpoch) break;
                    if (state.lastLoss < desiredLoss) break;

                    const result = await iterator.next();
                    if (result.done) break;
                    const batch = result.value;

                    this.trainBatch(state, batch);
                    const entry: TrainingLogEntry = {
                        epoch: state.epoch,
                        loss: state.lastLoss,
                        step: state.step,
                        time: Date.now() - startTime,
                        batchSize: batch.xs.shape[0],
                    };
                    this.model.log.push(entry);

                    if (state.step % logInterval === 0) {
                        if (onStep) {
                            if (prompt) {
                                const text = await generateText(this.model, prompt, 100, 0.8, 10);
                                entry.example = text;
                            }
                            await onStep(entry);
                        }
                    }
                }
            } catch (error) {
                console.error('Training error:', error);
                this.tf.dispose();
                throw error;
            }

            const avgLoss = state.epochLoss / state.step;

            // Validation
            if (validationDataset) {
                try {
                    const valLoss = await this.evaluateOnDataset(validationDataset, 5);
                    state.validationLosses.push(valLoss);

                    if (onEpoch) {
                        await onEpoch(state.epoch, avgLoss, valLoss);
                    }
                } catch (error) {
                    console.error('Validation error:', error);
                }
            } else {
                if (onEpoch) {
                    onEpoch(state.epoch, avgLoss);
                }
            }

            this.tf.dispose();

            if (state.lastLoss < desiredLoss) {
                break;
            }
        }

        return { losses: state.losses, validationLosses: state.validationLosses };
    }
}
