import { GPTConfig, LoRAConfig, validateConfig } from './models/config';
import type { ITokeniser } from './tokeniser/type';
import { saveModel, SaveOptions } from './loader/save';
import { loadModel, LoadModelOptions } from './loader/load';
import EE from 'eventemitter3';
import { dummyPassTrainAsync, MemoryRequirements } from './utilities/dummy';
import CharTokeniser from './tokeniser/CharTokeniser';
import { ConversationStream } from './data/stream';
import MemoryProfiler from './utilities/profile';
import BPETokeniser from './tokeniser/bpe';
import Model, { ModelForwardAttributes } from './models/model';
import createModelInstance from './models/factory';
import { ModelMode, TransformersMetadata } from './loader/types';
import Responses from './api/responses';
import Training from './api/training';
import { selectBackend } from './backend';
import { type GPUOptions, getBackendDevice } from './patches/webgpu_base';

type TeachableLLMStatus = 'warmup' | 'awaitingTokens' | 'ready' | 'training' | 'loading' | 'busy' | 'error';
type TeachableLLMEvents = 'status' | 'error' | 'loaded' | 'mode' | 'changeLoRA' | 'lost';

export default class TeachableLLM {
    static instances = new Set<TeachableLLM>();
    private ee = new EE<TeachableLLMEvents>();
    private _config?: GPTConfig;
    private _model?: Model<ModelForwardAttributes, GPTConfig>;
    private _tokeniser?: ITokeniser;
    private _status: TeachableLLMStatus = 'loading';
    private _memoryRequirements?: MemoryRequirements;
    private _responses: Responses | null = null;
    private _training: Training | null = null;
    public meta: TransformersMetadata = {
        version: 2,
        application: '@genai-fi/nanogpt',
    };

    static async selectBackend(backend: 'cpu' | 'webgl' | 'webgpu', options?: GPUOptions) {
        await selectBackend(backend, options);
        if (backend === 'webgpu') {
            const device = getBackendDevice();
            if (device) {
                device.lost.then(() => {
                    console.warn('WebGPU device lost');
                    TeachableLLM.instances.forEach((instance) => {
                        instance.setStatus('error');
                        instance.ee.emit('lost');
                    });
                });
            }
        }
    }

    constructor(tokeniser?: ITokeniser, model?: Model<ModelForwardAttributes, GPTConfig>) {
        this._config = model?.config;
        this._tokeniser = tokeniser;
        this._model = model;
        if (model?.metaData) {
            this.meta = model.metaData;
        }
        TeachableLLM.instances.add(this);
    }

    get vocab(): string[] {
        return this._tokeniser?.getVocab() || [];
    }

    get mode(): ModelMode {
        return this._model?.metaData?.mode ?? 'untrained';
    }

    set mode(mode: ModelMode) {
        if (!this._model) {
            throw new Error('model_not_initialized.');
        }

        if (this._model.metaData.mode === 'conversational' && mode === 'completion') {
            return;
        }
        if (mode === 'untrained') {
            return;
        }

        this._model.metaData.mode = mode;
        this.ee.emit('mode', mode);
    }

    /** Model is fully loaded */
    get loaded(): boolean {
        return !!this._model && !!this._tokeniser && !!this._config;
    }

    get config(): GPTConfig {
        if (!this._config) {
            throw new Error('configuration_not_initialized.');
        }
        return this._config;
    }

    get model(): Model<ModelForwardAttributes, GPTConfig> {
        if (!this._model) {
            throw new Error('model_not_initialized.');
        }
        return this._model;
    }

    get tokeniser(): ITokeniser {
        if (!this._tokeniser) {
            throw new Error('tokeniser_not_initialized.');
        }
        return this._tokeniser;
    }

    get status(): TeachableLLMStatus {
        return this._status;
    }

    /** Model is both ready and not busy */
    get ready(): boolean {
        return this._status === 'ready' && !!this._model && !!this._tokeniser;
    }

    get busy(): boolean {
        return this._status === 'busy' || this._status === 'training';
    }

    createLoRA(name: string, loraConfig: LoRAConfig) {
        if (!this._model) {
            throw new Error('model_not_initialized.');
        }
        this._model.createLoRA(name, loraConfig);
        this.ee.emit('changeLoRA');
    }

    deleteLoRA(name: string) {
        if (!this._model) {
            throw new Error('model_not_initialized.');
        }
        this._model.deleteLoRA(name);
        this.ee.emit('changeLoRA');
    }

    renameLoRA(oldName: string, newName: string) {
        if (!this._model) {
            throw new Error('model_not_initialized.');
        }
        this._model.renameLoRA(oldName, newName);
        this.ee.emit('changeLoRA');
    }

    attachLoRA(name: string) {
        if (!this._model) {
            throw new Error('model_not_initialized.');
        }
        if (this.model.lora?.name === name) {
            return; // Already attached
        }
        this._model.attachLoRA(name);
        this.ee.emit('changeLoRA');
    }

    detachLoRA() {
        if (!this._model) {
            throw new Error('model_not_initialized.');
        }
        this._model.detachLoRA();
        this.ee.emit('changeLoRA');
    }

    hasLoRA(name?: string): boolean {
        if (!this._model) {
            throw new Error('model_not_initialized.');
        }
        return this._model.hasLoRA(name);
    }

    listLoRAs(): string[] {
        if (!this._model) {
            throw new Error('model_not_initialized.');
        }
        return this._model.listLoRAs();
    }

    public estimateTrainingMemoryUsage(batchSize: number): number {
        const memReq = this._memoryRequirements ?? { perBatch: 0, tapeSize: 0, gradients: 0 };
        const batchMem = memReq.perBatch * batchSize;
        const gradientSize = memReq.gradients;
        return batchMem * 0.66 + gradientSize * 4;
    }

    private setStatus(status: TeachableLLMStatus) {
        if (this._status !== status) {
            this._status = status;
            this.ee.emit('status', status);
        }
    }

    saveModel(options?: SaveOptions): Promise<Blob> {
        if (!this._model || !this._tokeniser) {
            throw new Error('model_or_tokeniser_not_initialized.');
        }

        const trainingJob = (options?.includeOptimizer && this._training?.getPretrainingJob()) ?? null;

        return saveModel(
            this._model,
            this._tokeniser,
            {
                ...options,
                name: options?.name || this.meta.name,
            },
            trainingJob ? { optimizer: trainingJob.trainer.optimizer, trainingLog: trainingJob.trainer.log } : undefined
        );
    }

    static loadModel(data: Blob | Buffer | string, options?: LoadModelOptions): TeachableLLM {
        const teachableLLM = new TeachableLLM();
        loadModel(data, options)
            .then(({ model, tokeniser, metaData, optimizer, log }) => {
                validateConfig(model.config);
                teachableLLM._model = model;
                teachableLLM._tokeniser = tokeniser;
                teachableLLM._config = model.config;
                if (metaData) {
                    teachableLLM.meta = metaData;
                }
                teachableLLM.setStatus('warmup');

                dummyPassTrainAsync(model)
                    .then((memoryReqs) => {
                        teachableLLM._memoryRequirements = memoryReqs;

                        if (optimizer && model.metaData.pretrainingSettings && model.metaData.pretrainingData) {
                            teachableLLM.training.restore(
                                model.metaData.pretrainingSettings,
                                log || [],
                                optimizer,
                                model.metaData.pretrainingData
                            );
                        }

                        teachableLLM.setStatus('ready');
                        teachableLLM.ee.emit('loaded');
                        teachableLLM.ee.emit('mode', teachableLLM.mode);
                    })
                    .catch((err) => {
                        teachableLLM.setStatus('error');
                        teachableLLM.ee.emit('error', err);
                        console.error('Error during warmup:', err);
                    });
            })
            .catch((err) => {
                teachableLLM.setStatus('error');
                teachableLLM.ee.emit('error', err);
                console.error('Error loading model:', err);
            });

        return teachableLLM;
    }

    static create(tokeniserType: 'char' | 'bpe' | ITokeniser, config: GPTConfig) {
        validateConfig(config);
        const fullConfig = config;
        const tokeniser =
            tokeniserType === 'char'
                ? new CharTokeniser(fullConfig.vocabSize)
                : tokeniserType === 'bpe'
                  ? new BPETokeniser(fullConfig.vocabSize)
                  : tokeniserType;
        const model = createModelInstance(fullConfig);
        const tmodel = new TeachableLLM(tokeniser, model);
        tmodel.setStatus('warmup');

        dummyPassTrainAsync(model)
            .then((memoryReqs) => {
                tmodel._memoryRequirements = memoryReqs;

                if (tmodel.tokeniser.trained) {
                    tmodel.setStatus('ready');
                    tmodel.ee.emit('loaded');
                    tmodel.ee.emit('mode', tmodel.mode);
                } else {
                    tmodel.setStatus('awaitingTokens');
                    tmodel.ee.emit('loaded');
                    tmodel.ee.emit('mode', tmodel.mode);
                    tmodel.tokeniser.once('trainStatus', (status) => {
                        if (status === 'trained') {
                            tmodel.setStatus('ready');
                        }
                    });
                }
            })
            .catch((err) => {
                tmodel.setStatus('error');
                tmodel.ee.emit('error', err);
                console.error('Error during warmup:', err);
            });
        return tmodel;
    }

    getProfiler(): MemoryProfiler | undefined {
        return this._model?.getProfiler();
    }

    get enableProfiler(): boolean {
        return !!this._model?.getProfiler();
    }

    set enableProfiler(value: boolean) {
        if (value) {
            if (!this._config) {
                return;
            }
            if (!this.model.getProfiler()) {
                this.model.setProfiler(new MemoryProfiler());
            }
        } else {
            if (this.model.getProfiler()) {
                this.model.setProfiler(null);
            }
        }
    }

    getNumParams(): number {
        if (!this._model) {
            return 0;
        }
        return this._model.getNumParams();
    }

    async trainTokeniser(text: ConversationStream[]): Promise<number> {
        if (!this._tokeniser) {
            throw new Error('tokeniser_not_initialized.');
        }
        const tokenCount = await this._tokeniser.train(text);
        if (this._status === 'awaitingTokens') {
            this.setStatus('ready');
        }
        return tokenCount;
    }

    get responses() {
        if (!this._responses) {
            if (!this._model || !this._tokeniser) {
                throw new Error('model_or_tokeniser_not_initialized.');
            }
            this._responses = new Responses(this._model, this._tokeniser);
            this._responses.on('error', (error) => {
                this.setStatus('error');
                this.ee.emit('error', error);
            });
            this._responses.on('status', (status) => {
                if (status === 'busy') {
                    this.setStatus('busy');
                } else if (status === 'ready') {
                    this.setStatus('ready');
                }
            });
        }
        return this._responses;
    }

    get training() {
        if (!this._training) {
            if (!this._model || !this._tokeniser) {
                throw new Error('model_or_tokeniser_not_initialized.');
            }
            this._training = new Training(this._model, this._tokeniser);
            this._training.on('running', () => {
                this.setStatus('busy');
            });
            this._training.on('completed', () => {
                if (!this._training?.training) {
                    this.setStatus('ready');
                }
            });
            this._training.on('cancelled', () => {
                if (!this._training?.training) {
                    this.setStatus('ready');
                }
            });
            this._training.on('error', (error) => {
                this.setStatus('error');
                this.ee.emit('error', error);
            });
        }
        return this._training;
    }

    dispose() {
        if (this._responses) {
            this._responses.dispose();
            this._responses = null;
        }
        if (this._training) {
            this._training.dispose();
            this._training = null;
        }
        this._model?.dispose();
        this.ee.removeAllListeners();
        TeachableLLM.instances.delete(this);
    }

    on(event: 'status', listener: (status: TeachableLLMStatus) => void): void;
    on(event: 'mode', listener: (mode: ModelMode) => void): void;
    on(event: 'error', listener: (error: Error) => void): void;
    on(event: 'loaded' | 'changeLoRA', listener: () => void): void;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    on(event: TeachableLLMEvents, listener: (...args: any[]) => void): void {
        if (event === 'loaded' && this.loaded) {
            // If already loaded, call the listener immediately
            setTimeout(() => listener(), 0);
            return;
        }
        this.ee.on(event, listener);
    }

    off(event: 'status', listener: (status: TeachableLLMStatus) => void): void;
    off(event: 'mode', listener: (mode: ModelMode) => void): void;
    off(event: 'error', listener: (error: Error) => void): void;
    off(event: 'loaded' | 'changeLoRA', listener: () => void): void;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    off(event: TeachableLLMEvents, listener: (...args: any[]) => void): void {
        this.ee.off(event, listener);
    }
}
