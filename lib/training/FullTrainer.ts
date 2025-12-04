import type { ITokeniser } from '../tokeniser/type';
import GPTTrainer, { TrainingLogEntry, TrainingOptions, TrainingProgress } from './Trainer';
import Evaluator from './Evaluator';
import { dispose, Tensor } from '@tensorflow/tfjs-core';
import { Dataset } from '@tensorflow/tfjs-data';
import MemoryProfiler from '@base/utilities/profile';
import Model, { ModelForwardAttributes } from '@base/models/model';

interface TrainingState {
    step: number;
    lastLoss: number;
    totalSteps: number;
    losses: number[];
    validationLosses: number[];
    logStartTime: number;
    trainingDuration: number;
    //gradientNorm?: Promise<number>;
}

const DEFAULT_OPTIONS: TrainingOptions = {
    desiredLoss: 0.01,
    logInterval: 1,
    maxSteps: 1000,
};

// Enhanced training utilities with Dataset API and memory leak fixes
export default class FullTrainer extends GPTTrainer {
    constructor(model: Model<ModelForwardAttributes>, tokenizer: ITokeniser, learningRate: number = 3e-4) {
        super(model, tokenizer, learningRate);
    }

    private createEmptyState(): TrainingState {
        const state: TrainingState = {
            step: 0,
            lastLoss: 1e6,
            totalSteps: 0,
            losses: [],
            validationLosses: [],
            logStartTime: 0,
            trainingDuration: 0,
            ...(this.lastState || {}),
        };
        return state;
    }

    private createLogEntry(
        state: TrainingState,
        startTime: number,
        batchSize: number,
        advanced?: boolean
    ): TrainingLogEntry {
        const entry: TrainingLogEntry = {
            loss: state.lastLoss,
            step: state.step,
            time: Date.now() - startTime,
            batchSize: batchSize,
            learningRate: advanced ? this.optimizer.lr : undefined,
        };
        return entry;
    }

    private createProgress(state: TrainingState, entry: TrainingLogEntry, advanced?: boolean): TrainingProgress {
        const progress: TrainingProgress = {
            duration: state.trainingDuration,
            totalSamples: state.totalSteps * entry.batchSize,
            samplesPerSecond: (state.totalSteps * entry.batchSize) / (state.trainingDuration / 1000),
            memory: advanced ? this.model.getProfiler()?.getPeakMemory() || 0 : undefined,
        };
        return progress;
    }

    async stepDataset(
        dataset: Dataset<{ xs: Tensor; ys: Tensor }>,
        options: Partial<TrainingOptions>,
        validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>
    ): Promise<{ log: TrainingLogEntry; progress: TrainingProgress }> {
        const { logInterval } = {
            ...DEFAULT_OPTIONS,
            ...options,
        };

        const startTime = Date.now();

        const state = this.createEmptyState();
        this.lastState = state;

        await this.dummyPass();
        this.model.trainable = true;

        if (options?.advancedMetrics) {
            if (!this.model.getProfiler()) {
                this.model.setProfiler(new MemoryProfiler());
            }
        }

        this.running = true;
        state.logStartTime = startTime;

        const evaluator = validationDataset ? new Evaluator(this.model, validationDataset) : undefined;
        const iterator = await dataset.iterator();

        // Training loop with try-catch for better error handling
        try {
            while (this.running) {
                //if (state.lastLoss < desiredLoss) break;

                const result = await iterator.next();
                if (result.done) break;
                const batch = result.value;

                const lossScalar = this.trainBatch(state, batch);

                if (state.step % logInterval === 0) {
                    const lossValue = (await lossScalar.data())[0];
                    state.lastLoss = lossValue;
                    const logEndTime = Date.now();
                    state.trainingDuration += logEndTime - state.logStartTime;

                    const entry = this.createLogEntry(state, startTime, batch.xs.shape[0], options?.advancedMetrics);
                    this.model.trainingState = {
                        steps: state.totalSteps,
                        learningRate: this.optimizer.lr,
                        batchSize: batch.xs.shape[0],
                        loss: state.lastLoss,
                    };

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

                    const progress = this.createProgress(state, entry, options?.advancedMetrics);

                    lossScalar.dispose();
                    this.stop();
                    return { log: entry, progress };
                }
                lossScalar.dispose();
            }
        } catch (error) {
            console.error('Training error:', error);
            dispose();
            throw error;
        }

        dispose();

        this.running = false;

        throw new Error('No log returned before training stopped.');
    }

    // Train for multiple epochs using Dataset API - FIXED memory leaks
    async trainOnDataset(
        dataset: Dataset<{ xs: Tensor; ys: Tensor }>,
        options: Partial<TrainingOptions>,
        validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>
    ): Promise<{ losses: number[]; validationLosses: number[] }> {
        const { logInterval, onStep, maxSteps } = {
            ...DEFAULT_OPTIONS,
            ...options,
        };

        const startTime = Date.now();

        const state = this.createEmptyState();
        this.lastState = state;

        await this.dummyPass();
        this.model.trainable = true;

        if (options?.advancedMetrics) {
            if (!this.model.getProfiler()) {
                this.model.setProfiler(new MemoryProfiler());
            }
        }

        this.running = true;
        state.logStartTime = startTime;

        const evaluator = validationDataset ? new Evaluator(this.model, validationDataset) : undefined;
        const iterator = await dataset.iterator();

        // Training loop with try-catch for better error handling
        try {
            while (this.running) {
                //if (state.lastLoss < desiredLoss) break;

                const result = await iterator.next();
                if (result.done) break;
                const batch = result.value;

                const lossScalar = this.trainBatch(state, batch);

                if (state.step % logInterval === 0) {
                    const lossValue = (await lossScalar.data())[0];
                    state.lastLoss = lossValue;
                    const logEndTime = Date.now();
                    state.trainingDuration += logEndTime - state.logStartTime;

                    const entry = this.createLogEntry(state, startTime, batch.xs.shape[0], options?.advancedMetrics);
                    this.model.trainingState = {
                        steps: state.totalSteps,
                        learningRate: this.optimizer.lr,
                        batchSize: batch.xs.shape[0],
                        loss: state.lastLoss,
                    };

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
                        const progress = this.createProgress(state, entry, options?.advancedMetrics);

                        await onStep(entry, progress);
                    }

                    state.logStartTime = Date.now();
                }
                lossScalar.dispose();

                if (state.step >= maxSteps) {
                    this.stop();
                }
            }
        } catch (error) {
            console.error('Training error:', error);
            dispose();
            throw error;
        }

        dispose();

        this.running = false;

        return { losses: state.losses, validationLosses: state.validationLosses };
    }
}
