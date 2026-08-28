import EE from 'eventemitter3';
import type { ITokeniser } from '@base/tokeniser/type';
import Model, { type ModelForwardAttributes } from '@base/models/model';
import type { GPTConfig } from '@base/models/config';
import { ConversationStream } from '@base/data/stream';
import { TokenStore } from '@base/training/tasks/TokenStore';
import { DatasetMetadata } from '@base/loader/types';
import { TrainingLogEntry, TrainingOptions } from '@base/training/types';
import BasicTrainer from '@base/training/BasicTrainer';
import createTrainer from '@base/training/factory';
import { v4 as uuidv4 } from 'uuid';
import { Dataset } from '@tensorflow/tfjs-data';
import { Tensor } from '@tensorflow/tfjs-core';
import prepareData from '@base/training/prepareData';
import { AdamWOptimizer } from '@base/training/AdamW';
import generateDatasetID from '@base/utilities/datasetID';

export type TrainingState =
    | 'pending'
    | 'running'
    | 'paused'
    | 'pausing'
    | 'completed'
    | 'cancelled'
    | 'cancelling'
    | 'error';

export interface ITrainingJob {
    id: string;
    state: TrainingState;
    history: TrainingLogEntry[] | null;
    progress: number;
    remaining: number;
    trainer: BasicTrainer;
    options: TrainingOptions;
    trainDataset?: Dataset<{ xs: Tensor; ys: Tensor }>;
    validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>;
    totalTokens: number;
    datasets: DatasetMetadata[];
    datasetId?: string;
    breakOnLog: boolean;
}

interface TrainingEvents {
    error: (id: string, error: Error) => void;
    completed: (id: string) => void;
    running: (id: string) => void;
    paused: (id: string) => void;
    cancelled: (id: string) => void;
    pausing: (id: string) => void;
    cancelling: (id: string) => void;
    progress: (job: ITrainingJob) => void;
}

export default class Training {
    private ee: EE;
    // private _config?: GPTConfig;
    private _model: Model<ModelForwardAttributes, GPTConfig>;
    private _tokeniser: ITokeniser;
    private _jobs = new Map<string, ITrainingJob>();

    constructor(model: Model<ModelForwardAttributes, GPTConfig>, tokeniser: ITokeniser) {
        this.ee = new EE();
        this._model = model;
        this._tokeniser = tokeniser;
    }

    private setState(job: ITrainingJob, state: TrainingState, error?: Error) {
        const bad = () => {
            throw new Error(`invalid_state_transition_from_${job.state}_to_${state}`);
        };

        switch (state) {
            case 'pending':
                if (
                    job.state !== 'completed' &&
                    job.state !== 'paused' &&
                    job.state !== 'cancelled' &&
                    job.state !== 'pending'
                ) {
                    bad();
                }
                job.state = state;
                break;
            case 'running':
                if (job.state !== 'pending' && job.state !== 'paused') {
                    bad();
                }
                job.state = state;
                this.ee.emit('running', job.id);
                break;
            case 'paused':
                if (job.state !== 'pausing') {
                    bad();
                }
                job.state = state;
                this.ee.emit('paused', job.id);
                break;
            case 'completed':
                if (job.state !== 'running') {
                    bad();
                }
                job.state = state;
                this.ee.emit('completed', job.id);
                break;
            case 'pausing':
                if (job.state !== 'running') {
                    bad();
                }
                job.state = state;
                this.ee.emit('pausing', job.id);
                break;
            case 'cancelling':
                if (job.state !== 'running' && job.state !== 'paused' && job.state !== 'pausing') {
                    bad();
                }
                job.state = state;
                this.ee.emit('cancelling', job.id);
                break;
            case 'cancelled':
                if (job.state !== 'paused' && job.state !== 'cancelling') {
                    bad();
                }
                job.state = state;
                this.ee.emit('cancelled', job.id);
                break;
            case 'error':
                job.state = state;
                if (error) {
                    this.ee.emit('error', job.id, error);
                }
                break;
            default:
                throw new Error('invalid_state');
        }
    }

    get activeJobs() {
        return this._jobs.values().reduce((count, job) => (job.state === 'running' ? count + 1 : count), 0);
    }

    get training() {
        return this.activeJobs > 0;
    }

    private assertValidJobId(id: string): void {
        if (typeof id !== 'string' || id.trim().length === 0) {
            throw new Error('invalid_job_id');
        }
    }

    private assertValidOptions(options: TrainingOptions): void {
        if (!Number.isInteger(options.batchSize) || options.batchSize <= 0) {
            throw new Error('invalid_batch_size');
        }

        if (options.maxEpochs !== undefined && (!Number.isFinite(options.maxEpochs) || options.maxEpochs <= 0)) {
            throw new Error('invalid_max_epochs');
        }

        if (options.logInterval !== undefined && (!Number.isInteger(options.logInterval) || options.logInterval <= 0)) {
            throw new Error('invalid_log_interval');
        }

        if (
            options.learningRate !== undefined &&
            (!Number.isFinite(options.learningRate) || options.learningRate <= 0)
        ) {
            throw new Error('invalid_learning_rate');
        }

        if (
            options.validationSplit !== undefined &&
            (!Number.isFinite(options.validationSplit) || options.validationSplit <= 0 || options.validationSplit >= 1)
        ) {
            throw new Error('invalid_validation_split');
        }

        if (
            options.dropout !== undefined &&
            (!Number.isFinite(options.dropout) || options.dropout < 0 || options.dropout > 1)
        ) {
            throw new Error('invalid_dropout');
        }

        if (
            options.layerDrop !== undefined &&
            (!Number.isFinite(options.layerDrop) || options.layerDrop < 0 || options.layerDrop > 1)
        ) {
            throw new Error('invalid_layer_drop');
        }

        if (
            options.labelSmoothing !== undefined &&
            (!Number.isFinite(options.labelSmoothing) || options.labelSmoothing < 0 || options.labelSmoothing > 1)
        ) {
            throw new Error('invalid_label_smoothing');
        }
    }

    private assertValidDatasets(datasets: DatasetMetadata[]): void {
        if (datasets.length === 0) {
            throw new Error('invalid_datasets');
        }

        const seen = new Set<string>();
        for (const dataset of datasets) {
            if (!dataset.id.trim()) {
                throw new Error('invalid_dataset_id');
            }
            if (!dataset.name.trim()) {
                throw new Error('invalid_dataset_name');
            }
            if (seen.has(dataset.id)) {
                throw new Error('duplicate_dataset_id');
            }
            seen.add(dataset.id);
        }
    }

    private assertValidDataInput(data: ConversationStream[] | Uint16Array[] | TokenStore): void {
        if (data instanceof TokenStore) {
            if (data.tokeniserId !== this._tokeniser.id) {
                throw new Error('tokeniser_mismatch');
            }
            if (data.getTokenCount() === 0) {
                throw new Error('empty_training_data');
            }
            return;
        }

        if (data.length === 0) {
            throw new Error('empty_training_data');
        }
    }

    public on<E extends keyof TrainingEvents>(event: E, listener: TrainingEvents[E]): void {
        this.ee.on(event, listener);
    }

    public off<E extends keyof TrainingEvents>(event: E, listener: TrainingEvents[E]): void {
        this.ee.off(event, listener);
    }

    public restore(
        options: TrainingOptions,
        log: TrainingLogEntry[],
        optimizer: AdamWOptimizer,
        datasets: DatasetMetadata[]
    ): ITrainingJob {
        this.assertValidOptions(options);
        this.assertValidDatasets(datasets);
        // TODO: Check the optimizer is compatible with the model and options.

        if (log.length === 0) {
            throw new Error('invalid_log');
        }

        const job: ITrainingJob = {
            id: uuidv4(),
            state: 'completed',
            trainer: createTrainer(this._model, this._tokeniser, options, optimizer),
            history: log,
            progress: log[log.length - 1]?.totalTokens / log[log.length - 1]?.duration || 0,
            remaining: 0,
            options,
            totalTokens: log[log.length - 1]?.totalTokens || 0,
            datasets,
            datasetId: generateDatasetID(datasets),
            breakOnLog: false,
        };

        job.trainer.log = log;
        job.trainer.resumeFromLog(log[log.length - 1]);

        this._jobs.set(job.id, job);
        return job;
    }

    private launchJob(job: ITrainingJob) {
        if (!job.trainDataset) return;

        this.setState(job, 'running');

        job.trainer
            .trainOnDataset(job.trainDataset, job.options, job.validationDataset, (log: TrainingLogEntry) => {
                job.history = job.trainer.log;
                job.progress = log.totalTokens / job.totalTokens;
                job.remaining = Math.max(0, ((job.totalTokens - log.totalTokens) / log.totalTokens) * log.duration);
                if (job.breakOnLog) {
                    if (job.state === 'running') {
                        this.setState(job, 'pausing');
                        job.trainer.stop();
                    }
                }
                this.ee.emit('progress', job);
            })
            .then(() => {
                if (job.state === 'running') {
                    this.setState(job, 'completed');
                } else if (job.state === 'pausing') {
                    this.setState(job, 'paused');
                } else if (job.state === 'cancelling') {
                    this.setState(job, 'cancelled');
                }
            })
            .catch((error) => {
                this.setState(job, 'error', error);
            });
    }

    public async job(
        options: TrainingOptions,
        data: ConversationStream[] | Uint16Array[] | TokenStore = [],
        datasets: DatasetMetadata[],
        validation?: Uint16Array[] | TokenStore
    ): Promise<ITrainingJob> {
        this.assertValidOptions(options);
        this.assertValidDataInput(data);
        this.assertValidDatasets(datasets);

        if (validation && validation instanceof TokenStore && validation.tokeniserId !== this._tokeniser.id) {
            throw new Error('tokeniser_mismatch');
        }

        // Cleanup any completed jobs of the same type.
        const jobIds = Array.from(this._jobs.keys());
        for (const id of jobIds) {
            const job = this._jobs.get(id);
            if (!job) continue;
            if (job.id === options.previous_job_id) {
                continue;
            }
            if (job.options.method.type === options.method.type && job.state !== 'running') {
                job.trainer.dispose();
                this._jobs.delete(job.id);
            } else if (job.options.method.type === options.method.type && job.state === 'running') {
                throw new Error('training_in_progress');
            }
        }

        const job: ITrainingJob | null = options.previous_job_id
            ? this.getJob(options.previous_job_id)
            : {
                  id: uuidv4(),
                  state: 'pending',
                  trainer: createTrainer(this._model, this._tokeniser, options),
                  history: [],
                  progress: 0,
                  remaining: 0,
                  options,
                  totalTokens: 0,
                  datasets,
                  breakOnLog: false,
              };

        if (!job) {
            throw new Error('invalid_previous_job_id');
        }

        this._jobs.set(job.id, job);

        this.setState(job, 'pending');

        if (data instanceof TokenStore) {
            if (job.datasetId && data.datasetId !== job.datasetId) {
                const err = new Error('dataset_mismatch');
                this.setState(job, 'error', err);
                throw err;
            }
            if (data.tokeniserId !== this._tokeniser.id) {
                const err = new Error('tokeniser_mismatch');
                this.setState(job, 'error', err);
                throw err;
            }
            job.datasetId = data.datasetId;
        }

        if (!job.trainDataset) {
            try {
                const preparedData = await prepareData(
                    options,
                    this._model,
                    this._tokeniser,
                    data,
                    job.datasetId ?? generateDatasetID(datasets),
                    validation,
                    datasets
                );
                job.trainDataset = preparedData.trainDataset;
                job.validationDataset = preparedData.validationDataset;
                job.totalTokens = preparedData.totalTokens;
            } catch (error) {
                this.setState(job, 'error', new Error('prepare_data_failed'));
                job.trainer.dispose();
                this._jobs.delete(job.id);
                throw error;
            }
        }

        if (!options.previous_job_id) {
            try {
                job.trainer.configure(options);
            } catch (error) {
                this.setState(job, 'error', new Error('trainer_configuration_failed'));
                job.trainer.dispose();
                this._jobs.delete(job.id);
                throw error;
            }
        } else {
            this.assertValidOptions(options);
            job.options = options;
            try {
                job.trainer.configure(options);
            } catch (error) {
                this.setState(job, 'error', new Error('trainer_configuration_failed'));
                job.trainer.dispose();
                this._jobs.delete(job.id);
                throw error;
            }
        }

        this.launchJob(job);

        return job;
    }

    public getJob(id: string): ITrainingJob | null {
        this.assertValidJobId(id);
        return this._jobs.get(id) ?? null;
    }

    /** Resume a paused job and optionally change some options. */
    public async resume(id: string) {
        this.assertValidJobId(id);

        const job = this.getJob(id);
        if (!job) return;

        if (job.state !== 'paused') {
            return;
        }

        this.setState(job, 'pending');
        this.launchJob(job);
    }

    public cancel(id: string) {
        this.assertValidJobId(id);
        const job = this.getJob(id);
        if (!job) return;

        if (job.state === 'paused') {
            this.setState(job, 'cancelled');
            return;
        } else if (job.state === 'running' || job.state === 'pausing') {
            this.setState(job, 'cancelling');
            job.trainer.stop();
        }
    }

    public breakpoints(id: string, enabled: boolean) {
        const job = this.getJob(id);
        if (job) {
            job.breakOnLog = enabled;
        }
    }

    public getPretrainingJob(): ITrainingJob | null {
        for (const job of this._jobs.values()) {
            if (job.options.method.type === 'pretraining') {
                return job;
            }
        }
        return null;
    }

    dispose() {
        for (const job of this._jobs.values()) {
            if (job.state === 'running') {
                throw new Error('job_still_running');
            }
            job.trainer.dispose();
        }
        this._jobs.clear();
        this.ee.removeAllListeners();
    }
}
