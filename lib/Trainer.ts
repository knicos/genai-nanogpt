import type { ITokeniser } from './tokeniser/type';
import EE from 'eventemitter3';
import FullTrainer from './training/FullTrainer';
import { TrainingLogEntry, TrainingProgress } from './training/Trainer';
import { Dataset } from '@tensorflow/tfjs-data';
import { Tensor } from '@tensorflow/tfjs-core';
import Model, { ModelForwardAttributes } from './models/model';
import { Task } from './training/tasks/Task';

export interface ITrainerOptions {
    batchSize?: number; // Batch size for training
    learningRate?: number; // Learning rate for the optimizer
    maxSteps?: number; // Maximum training steps
    desiredLoss?: number; // Desired loss to stop training
    logInterval?: number; // Interval for logging training progress
    prompt?: string; // Prompt for generating text during training
    validationSplit?: number; // Fraction of data to use for validation
    advancedMetrics?: boolean; // Whether to compute advanced metrics during training
    gradientCheckpointing?: boolean; // Whether to use gradient checkpointing
    gradientMetrics?: boolean; // Whether to compute gradient metrics during training
    mixedPrecision?: boolean; // Whether to use mixed precision training
}

interface ExtendedTrainingProgress extends TrainingProgress {
    progress: number; // Progress as a fraction between 0 and 1
    remaining: number; // Estimated remaining time in seconds
}

export default class Trainer extends EE<'start' | 'stop' | 'log'> {
    private trainer: FullTrainer;
    private hasTrained: boolean = false;
    private trainDataset?: Dataset<{ xs: Tensor; ys: Tensor }>;
    private validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>;
    private totalSamples: number = 0;
    private log: TrainingLogEntry[] = [];
    private progress: ExtendedTrainingProgress | null = null;

    constructor(model: Model<ModelForwardAttributes>, tokeniser: ITokeniser) {
        super();
        this.trainer = new FullTrainer(model, tokeniser, 1e-3);
    }

    stop() {
        this.trainer.stop();
    }

    reset() {
        this.hasTrained = false;
        this.log = [];
        this.trainer.reset();
    }

    getTotalSamples(): number {
        return this.totalSamples;
    }

    async prepare(tasks: Task[], options?: ITrainerOptions): Promise<void> {
        const { trainDataset, validationDataset, size } = await this.trainer.createTrainValidationSplit(
            tasks,
            options?.batchSize || 32,
            options?.validationSplit || 0.1
        );

        const totalSamples = size * (1 - (options?.validationSplit || 0));

        this.trainDataset = trainDataset;
        this.validationDataset = validationDataset;
        this.totalSamples = totalSamples;
    }

    async train(options?: ITrainerOptions): Promise<void> {
        if (!this.trainDataset || !this.validationDataset) {
            throw new Error('Datasets not prepared');
        }

        // Only set the learning rate if we haven't trained before
        // This allows for resuming training without resetting the learning rate
        if (!this.hasTrained) {
            this.trainer.setLearningRate(options?.learningRate || 1e-3);
        }
        this.hasTrained = true;

        this.emit('start');

        this.trainer.setGradientCheckpointing(options?.gradientCheckpointing || false);
        this.trainer.setMixedPrecision(options?.mixedPrecision || false);

        await this.trainer.trainOnDataset(
            this.trainDataset,
            {
                prompt: options?.prompt,
                logInterval: options?.logInterval || 10,
                desiredLoss: options?.desiredLoss || 0.01,
                maxSteps: options?.maxSteps || 1000,
                advancedMetrics: options?.advancedMetrics || false,
                gradientMetrics: options?.gradientMetrics || false,
                onStep: async (log: TrainingLogEntry, progress: TrainingProgress) => {
                    this.log.push(log);
                    this.progress = {
                        ...progress,
                        progress: progress.totalSamples / this.totalSamples,
                        remaining: Math.max(
                            0,
                            ((this.totalSamples - progress.totalSamples) / progress.totalSamples) * progress.duration
                        ),
                    };

                    const listeners = this.listeners('log');
                    for (const listener of listeners) {
                        // These listeners can be async, so we await them
                        await listener(log, this.progress!);
                    }
                },
            },
            this.validationDataset
        );
        this.emit('stop');
    }

    async step(options?: ITrainerOptions): Promise<void> {
        if (!this.trainDataset || !this.validationDataset) {
            throw new Error('Datasets not prepared');
        }
        // Only set the learning rate if we haven't trained before
        // This allows for resuming training without resetting the learning rate
        if (!this.hasTrained) {
            this.trainer.setLearningRate(options?.learningRate || 1e-3);
        }
        this.hasTrained = true;

        this.emit('start');

        const { log, progress } = await this.trainer.stepDataset(
            this.trainDataset,
            {
                prompt: options?.prompt,
                logInterval: options?.logInterval || 10,
                desiredLoss: options?.desiredLoss || 0.01,
                maxSteps: options?.maxSteps || 1000,
                advancedMetrics: options?.advancedMetrics || false,
            },
            this.validationDataset
        );

        const listeners = this.listeners('log');
        for (const listener of listeners) {
            // These listeners can be async, so we await them
            await listener(log, {
                ...progress,
                progress: progress.totalSamples / this.totalSamples,
                remaining: Math.max(
                    0,
                    ((this.totalSamples - progress.totalSamples) / progress.totalSamples) * progress.duration
                ),
            });
        }
        this.emit('stop');
    }

    getLog(): TrainingLogEntry[] {
        return this.log;
    }

    getProgress(): ExtendedTrainingProgress | null {
        return this.progress;
    }

    isPrepared(): boolean {
        return this.trainDataset !== undefined && this.validationDataset !== undefined;
    }
}
