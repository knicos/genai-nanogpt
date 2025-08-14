import { ITokeniser } from '../tokeniser/type';
import { generateText } from '../utilities/generate';
import NanoGPT, { TrainingLogEntry } from '../NanoGPTModel';
import type TF from '@tensorflow/tfjs';
import GPTTrainer, { TrainingOptions } from './Trainer';
import Evaluator from './Evaluator';

interface TrainingState {
    step: number;
    lastLoss: number;
    totalSteps: number;
    losses: number[];
    validationLosses: number[];
}

const DEFAULT_OPTIONS: TrainingOptions = {
    desiredLoss: 0.01,
    logInterval: 1,
    maxSteps: 1000,
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
        const { desiredLoss, logInterval, onStep, prompt, maxSteps } = {
            ...DEFAULT_OPTIONS,
            ...options,
        };

        const state: TrainingState = {
            step: 0,
            lastLoss: 1e6,
            totalSteps: 0,
            losses: [],
            validationLosses: [],
            ...(this.lastState || {}),
        };
        this.lastState = state;

        console.log('Starting training with options:');
        this.dummyPass();
        console.log('Dummy pass complete');
        this.model.trainable = true;

        const startTime = Date.now();

        this.running = true;

        const evaluator = validationDataset ? new Evaluator(this.model, validationDataset) : undefined;

        const iterator = await dataset.iterator();

        // Training loop with try-catch for better error handling
        try {
            while (this.running) {
                if (state.lastLoss < desiredLoss) break;

                const result = await iterator.next();
                if (result.done) break;
                const batch = result.value;

                const lossPromise = this.trainBatch(state, batch);

                const entry: TrainingLogEntry = {
                    loss: state.lastLoss,
                    step: state.step,
                    time: Date.now() - startTime,
                    batchSize: batch.xs.shape[0],
                };
                this.model.log.push(entry);

                if (state.step % logInterval === 0) {
                    await lossPromise;
                    // Validation
                    if (evaluator) {
                        try {
                            const valLoss = await evaluator.evaluate(5);
                            state.validationLosses.push(valLoss);
                            entry.valLoss = valLoss;
                        } catch (error) {
                            console.error('Validation error:', error);
                        }
                    }
                    if (onStep) {
                        if (prompt) {
                            const text = await generateText(this.tokenizer, this.model, prompt, 100, {
                                temperature: 0.8,
                            });
                            entry.example = text;
                        }
                        await onStep(entry);
                    }
                }

                if (state.step >= maxSteps) {
                    this.stop();
                }
            }
        } catch (error) {
            console.error('Training error:', error);
            this.tf.dispose();
            throw error;
        }

        this.tf.dispose();

        this.running = false;

        return { losses: state.losses, validationLosses: state.validationLosses };
    }
}
