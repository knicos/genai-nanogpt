import type { ITokeniser } from './tokeniser/type';
import EE from 'eventemitter3';
import PreTrainer from './training/PreTrainer';
import { Dataset } from '@tensorflow/tfjs-data';
import { Tensor } from '@tensorflow/tfjs-core';
import Model, { ModelForwardAttributes } from './models/model';
import { Task } from './training/tasks/Task';
import { TrainingOptions, TrainingLogEntry } from './training/types';
import { createTrainValidationSplit } from './training/validation';
import SFTTrainer from './training/SFTTrainer';
import { AdamWOptimizer } from './training/AdamW';
import splitValidation from './training/tasks/splitter';
import { v4 as uuidv4 } from 'uuid';
import { DatasetMetadata } from './loader/types';

interface TrainingProgress {
    lastLog: TrainingLogEntry;
    progress: number; // Progress as a fraction between 0 and 1
    remaining: number; // Estimated remaining time in seconds
}

export type TrainingType = 'pretraining' | 'sft';

export default class Trainer extends EE<'start' | 'stop' | 'log'> {
    private trainer: PreTrainer | SFTTrainer;
    public readonly trainingType: TrainingType = 'pretraining';
    private hasTrained = false;
    private trainDataset?: Dataset<{ xs: Tensor; ys: Tensor }>;
    private validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>;
    private totalTokens = 0;
    private tokensProcessed = 0;
    public log: TrainingLogEntry[] = [];
    private progress: TrainingProgress | null = null;
    public options: TrainingOptions = {
        batchSize: 32,
        sftMode: 'full',
        logInterval: 10,
    };
    protected tokenizer: ITokeniser;

    constructor(
        model: Model<ModelForwardAttributes>,
        tokeniser: ITokeniser,
        trainingType?: TrainingType,
        options?: TrainingOptions,
        optimizer?: AdamWOptimizer
    );
    constructor(trainer: Trainer, options?: TrainingOptions);
    constructor(
        modelOrCopy: Model<ModelForwardAttributes> | Trainer,
        tokeniser?: ITokeniser | TrainingOptions,
        trainingType?: TrainingType,
        options?: TrainingOptions,
        optimizer?: AdamWOptimizer
    ) {
        super();

        if (modelOrCopy instanceof Trainer) {
            const newOptions = tokeniser ? (tokeniser as TrainingOptions) : modelOrCopy.options;
            const oldOptions = modelOrCopy.options;

            let needsReset = false;
            if (modelOrCopy.trainingType === 'sft' && newOptions.sftMode !== oldOptions.sftMode) {
                needsReset = true;
            }
            if (
                modelOrCopy.trainer instanceof SFTTrainer &&
                modelOrCopy.trainer.loraName &&
                newOptions.loraName !== modelOrCopy.trainer.loraName
            ) {
                needsReset = true;
            }
            if (trainingType !== undefined && trainingType !== modelOrCopy.trainingType) {
                needsReset = true;
            }

            // Sometimes need to get a new optimizer
            if (needsReset) {
                if (modelOrCopy.trainingType === 'sft') {
                    this.trainer = new SFTTrainer(modelOrCopy.model, modelOrCopy.tokenizer, newOptions);
                    this.trainer.loraName = newOptions.loraName;
                } else {
                    this.trainer = new PreTrainer(modelOrCopy.model, modelOrCopy.tokenizer, newOptions);
                }
                this.trainingType = trainingType || modelOrCopy.trainingType;
                this.options = newOptions;
                this.tokenizer = modelOrCopy.tokenizer;
            } else {
                this.trainer = modelOrCopy.trainer;
                this.trainingType = trainingType || modelOrCopy.trainingType;
                this.options = newOptions;
                this.trainer.updateOptimizer(this.options);
                this.log = modelOrCopy.log;
                this.progress = modelOrCopy.progress;
                this.totalTokens = modelOrCopy.totalTokens;
                this.tokenizer = modelOrCopy.tokenizer;

                if (newOptions.batchSize === oldOptions.batchSize) {
                    this.trainDataset = modelOrCopy.trainDataset;
                    this.validationDataset = modelOrCopy.validationDataset;
                }
            }
            return;
        }

        if (!tokeniser) {
            throw new Error('Tokeniser must be provided when initializing Trainer with a model');
        }
        if (!modelOrCopy) {
            throw new Error('Model must be provided when initializing Trainer');
        }

        this.options = options || {
            batchSize: 32,
            sftMode: 'full',
        };
        if (trainingType === 'sft') {
            this.trainer = new SFTTrainer(modelOrCopy, tokeniser as ITokeniser, options, optimizer);
            this.trainer.loraName = options?.loraName;
        } else {
            this.trainer = new PreTrainer(modelOrCopy, tokeniser as ITokeniser, options, optimizer);
        }
        this.trainingType = trainingType || 'pretraining';
        this.tokenizer = tokeniser as ITokeniser;
    }

    get model(): Model<ModelForwardAttributes> {
        return this.trainer.model;
    }

    get optimizer(): AdamWOptimizer {
        return this.trainer.optimizer;
    }

    stop() {
        this.trainer.stop();
    }

    reset() {
        this.hasTrained = false;
        this.log = [];
        this.trainer.reset();
    }

    dispose() {
        this.trainer.dispose();
        this.removeAllListeners();
    }

    getTotalTokens(): number {
        return this.totalTokens;
    }

    setOptions(options: TrainingOptions): void {
        // Check which options have changed and only update those
        // This allows us to change options like learning rate or metrics on the fly without resetting the entire trainer
        // Throw if some options are changed during training such as batchSize.

        const changedOptions = new Set(
            Object.keys(options).filter(
                (key) => options[key as keyof TrainingOptions] !== this.options[key as keyof TrainingOptions]
            )
        );

        if (this.trainer.isRunning) {
            if (changedOptions.has('batchSize')) {
                throw new Error('Cannot change batch size during training');
            }
            if (changedOptions.has('sftMode')) {
                throw new Error('Cannot change SFT mode during training');
            }
            if (changedOptions.has('loraConfig')) {
                throw new Error('Cannot change LoRA configuration during training');
            }
            if (changedOptions.has('validationSplit')) {
                throw new Error('Cannot change validation split during training');
            }
            if (changedOptions.has('trainableWeights')) {
                throw new Error('Cannot change trainable weights during training');
            }
            if (changedOptions.has('mixedPrecision')) {
                throw new Error('Cannot change mixed precision setting during training');
            }
            if (changedOptions.has('gradientCheckpointing')) {
                throw new Error('Cannot change gradient checkpointing setting during training');
            }
        }

        this.options = {
            ...this.options,
            ...options,
        };

        this.trainer.updateOptimizer(this.options);
        if (changedOptions.has('metrics')) {
            this.trainer.setMetrics(options.metrics || []);
        }
    }

    async prepare(tasks: Task[] | Uint16Array = [], datasets?: DatasetMetadata[]): Promise<void> {
        const options = this.options;

        if (datasets) {
            this.model.metaData.pretrainingData = datasets.map((dataset) => ({
                id: dataset.id,
                name: dataset.name,
            }));
        }

        if (this.trainingType === 'pretraining' && this.trainer instanceof PreTrainer) {
            const { trainDataset, validationDataset, size } = await createTrainValidationSplit(
                tasks,
                this.trainer.tokenizer,
                this.trainer.datasetBuilder,
                options?.batchSize || 32,
                options?.validationSplit || 0.1
            );

            const totalTokens = size * (1 - (options?.validationSplit || 0));

            this.trainDataset = trainDataset;
            this.validationDataset = validationDataset;
            this.totalTokens = totalTokens;
            this.options.epochSteps = Math.ceil(
                this.totalTokens / ((options?.batchSize || 32) * this.model.config.blockSize)
            );
            this.trainer.updateOptimizer(this.options);
        } else if (this.trainingType === 'sft' && this.trainer instanceof SFTTrainer) {
            if (tasks instanceof Uint16Array) {
                throw new Error('SFT training requires Task[] input');
            }

            if (options?.validationSplit && options.validationSplit > 0) {
                const splitTasks = splitValidation(tasks, options?.validationSplit);

                const trainDataset = await this.trainer.datasetBuilder.createSFTDataset(
                    [splitTasks.training],
                    options?.batchSize || 32,
                    -100
                );
                const validationDataset = await this.trainer.datasetBuilder.createSFTDataset(
                    [splitTasks.validation],
                    options?.batchSize || 32,
                    -100
                );

                this.validationDataset = validationDataset;
                this.trainDataset = trainDataset;
            } else {
                const trainDataset = await this.trainer.datasetBuilder.createSFTDataset(
                    tasks,
                    options?.batchSize || 32,
                    -100
                );

                this.trainDataset = trainDataset;
            }
            this.totalTokens = tasks.reduce((acc, conv) => acc + conv.length, 0);
            this.options.epochSteps = Math.ceil(
                this.totalTokens / ((options?.batchSize || 32) * this.model.config.blockSize)
            );
            this.trainer.updateOptimizer(this.options);
        }
    }

    private configureModel(options?: TrainingOptions) {
        const mode = options?.sftMode || 'full';

        if (this.trainingType === 'pretraining') {
            if (this.trainer.model.hasLoRA()) {
                this.trainer.model.detachLoRA();
            }
            this.trainer.model.weightStore.setTrainable(['*']);
        }

        if (this.trainingType === 'sft') {
            if (mode === 'lora') {
                const model = this.trainer.model;

                console.log('Configuring model for LoRA fine-tuning with options:', options);

                if (options?.loraName) {
                    if (!model.hasLoRA(options.loraName)) {
                        if (options.loraConfig) {
                            model.createLoRA(options.loraName, options.loraConfig);
                            model.attachLoRA(options.loraName);
                        } else {
                            throw new Error(
                                `LoRA configuration must be provided to create LoRA with name ${options.loraName}`
                            );
                        }
                    } else {
                        model.attachLoRA(options.loraName);
                        if (options.loraConfig) {
                            const existingLoRA = model.lora!;
                            if (
                                existingLoRA.alpha !== options.loraConfig.alpha ||
                                existingLoRA.rank !== options.loraConfig.rank
                            ) {
                                // Reset to a new LoRA
                                model.detachLoRA();
                                model.deleteLoRA(options.loraName);
                                model.createLoRA(options.loraName, options.loraConfig);
                                model.attachLoRA(options.loraName);
                                console.warn('Resetting LoRA with new configuration.');
                            }
                        }
                    }
                } else if (options?.loraConfig) {
                    if (model.hasLoRA()) {
                        const existingLoRA = model.lora!;
                        if (
                            existingLoRA.alpha !== options.loraConfig.alpha ||
                            existingLoRA.rank !== options.loraConfig.rank
                        ) {
                            // Reset to a new LoRA
                            model.detachLoRA();
                            const loraName = options.loraName || uuidv4();
                            model.createLoRA(loraName, options.loraConfig);
                            model.attachLoRA(loraName);
                        }
                    } else {
                        const loraName = options.loraName || uuidv4();
                        model.createLoRA(loraName, options.loraConfig);
                        model.attachLoRA(loraName);
                    }
                } else if (model.hasLoRA()) {
                    // Keep existing LoRA
                } else {
                    throw new Error('LoRA configuration must be provided for lora SFT mode');
                }
            } else {
                if (this.trainer.model.hasLoRA()) {
                    this.trainer.model.detachLoRA();
                }
            }

            if (mode === 'last-layer') {
                this.trainer.model.weightStore.setTrainable([
                    `block_${this.trainer.model.config.nLayer - 1}_*`,
                    'token_embedding',
                ]);
            } else if (mode === 'full') {
                this.trainer.model.weightStore.setTrainable(['*']);
            }
        }

        if (options?.trainableWeights) {
            this.trainer.model.weightStore.setTrainable(options.trainableWeights);
        }
    }

    async train(): Promise<void> {
        const options = this.options;

        if (!this.trainDataset) {
            throw new Error('Dataset not prepared');
        }

        // Only set the learning rate if we haven't trained before
        // This allows for resuming training without resetting the learning rate
        if (!this.hasTrained) {
            this.trainer.setLearningRate(options?.learningRate || 1e-3);
        }
        this.hasTrained = true;

        this.emit('start');

        this.model.metaData.pretrainingSettings = options;
        const startTime = Date.now();

        // Ensure trainer state is resumed
        if (this.log.length > 0) {
            this.trainer.resumeFromLog(this.log[this.log.length - 1]);
        }

        this.trainer.setGradientCheckpointing(options?.gradientCheckpointing || false);
        this.trainer.setMixedPrecision(options?.mixedPrecision || false);
        this.trainer.setLabelSmoothing(options?.labelSmoothing || 0.0);
        this.trainer.setDropout(options?.dropout || 0);
        this.trainer.setLayerDrop(options?.layerDrop || 0);
        this.configureModel(options);

        await this.trainer.trainOnDataset(
            this.trainDataset,
            {
                ...options,
                onStep: async (log: TrainingLogEntry) => {
                    this.log.push(log);
                    this.progress = {
                        lastLog: log,
                        progress: log.totalTokens / this.totalTokens,
                        remaining: Math.max(0, ((this.totalTokens - log.totalTokens) / log.totalTokens) * log.duration),
                    };
                    this.tokensProcessed = log.totalTokens;

                    const listeners = this.listeners('log');
                    for (const listener of listeners) {
                        // These listeners can be async, so we await them
                        await listener(log, this.progress!);
                    }
                },
            },
            this.validationDataset
        );

        this.model.metaData.actionLog = this.model.metaData.actionLog || [];
        const endTime = Date.now();
        this.model.metaData.actionLog.push({
            action: 'pretrain',
            timestamp: endTime,
            duration: endTime - startTime,
            tokensProcessed: this.tokensProcessed,
            options,
        });

        this.emit('stop');
    }

    async step(options?: TrainingOptions): Promise<void> {
        if (!this.trainDataset) {
            throw new Error('Dataset not prepared');
        }
        // Only set the learning rate if we haven't trained before
        // This allows for resuming training without resetting the learning rate
        if (!this.hasTrained) {
            this.trainer.setLearningRate(options?.learningRate || 1e-3);
        }
        this.hasTrained = true;

        this.emit('start');

        const { log } = await this.trainer.stepDataset(this.trainDataset, options || {}, this.validationDataset);

        const listeners = this.listeners('log');
        for (const listener of listeners) {
            // These listeners can be async, so we await them
            await listener(log, {
                lastLog: log,
                progress: log.totalTokens / this.totalTokens,
                remaining: Math.max(0, ((this.totalTokens - log.totalTokens) / log.totalTokens) * log.duration),
            });
        }
        this.emit('stop');
    }

    getLog(): TrainingLogEntry[] {
        return this.log;
    }

    getProgress(): TrainingProgress | null {
        return this.progress;
    }

    isPrepared(): boolean {
        return this.trainDataset !== undefined && this.validationDataset !== undefined;
    }
}
