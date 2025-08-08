import NanoGPT, { TrainingLogEntry } from './NanoGPTModel';
import { ITokeniser } from './tokeniser/type';
import EE from 'eventemitter3';
import FullTrainer from './training/FullTrainer';

export interface ITrainerOptions {
    batchSize?: number; // Batch size for training
    learningRate?: number; // Learning rate for the optimizer
    maxSteps?: number; // Maximum training steps
    desiredLoss?: number; // Desired loss to stop training
    logInterval?: number; // Interval for logging training progress
    prompt?: string; // Prompt for generating text during training
    validationSplit?: number; // Fraction of data to use for validation
}

export default class Trainer extends EE<'start' | 'stop' | 'log'> {
    private trainer: FullTrainer;

    constructor(model: NanoGPT, tokeniser: ITokeniser) {
        super();
        this.trainer = new FullTrainer(model.tf, model, tokeniser, 1e-3);
    }

    stop() {}

    async train(text: string[], options?: ITrainerOptions): Promise<void> {
        const { trainDataset, validationDataset } = await this.trainer.createTrainValidationSplit(
            text,
            options?.batchSize || 32,
            options?.validationSplit || 0.1
        );
        this.emit('start');
        await this.trainer.trainOnDataset(
            trainDataset,
            {
                prompt: options?.prompt,
                logInterval: options?.logInterval || 10,
                desiredLoss: options?.desiredLoss || 0.01,
                onStep: async (log: TrainingLogEntry) => {
                    const listeners = this.listeners('log');
                    for (const listener of listeners) {
                        // These listeners can be async, so we await them
                        await listener(log);
                    }
                },
            },
            validationDataset
        );
        this.emit('stop');
    }
}
