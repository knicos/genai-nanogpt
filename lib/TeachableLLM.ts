import { defaultConfig, GPTConfig } from './config';
import type { ITokeniser } from './tokeniser/type';
import NanoGPT, { TrainingLogEntry } from './NanoGPTModel';
import { saveModel, SaveOptions } from './utilities/save';
import { loadModel } from './loader/load';
import Generator, { IGenerateOptions } from './Generator';
import Trainer, { ITrainerOptions } from './Trainer';
import EE from 'eventemitter3';
import { dummyPassTrainAsync, MemoryRequirements } from './utilities/dummy';
import { CharTokeniser } from './main';
import MemoryProfiler from './utilities/profile';
import BPETokeniser from './tokeniser/bpe';
import { TrainingProgress } from './training/Trainer';
import { GPTLayerConfig } from './layers/BaseLayer';

type TeachableLLMStatus = 'warmup' | 'awaitingTokens' | 'ready' | 'training' | 'loading' | 'busy' | 'error';
type TeachableLLMEvents = 'status' | 'error' | 'trainStep' | 'loaded';

interface TeachableLLMMeta {
    name?: string;
    id?: string;
    [key: string]: unknown;
}

export default class TeachableLLM {
    private ee = new EE<TeachableLLMEvents>();
    private _config?: GPTLayerConfig;
    private _model?: NanoGPT;
    private _tokeniser?: ITokeniser;
    private _status: TeachableLLMStatus = 'loading';
    private _memoryRequirements?: MemoryRequirements;
    public meta: TeachableLLMMeta = {};

    constructor(tokeniser?: ITokeniser, model?: NanoGPT) {
        this._config = model?.config;
        this._tokeniser = tokeniser;
        this._model = model;
    }

    get vocab(): string[] {
        return this._tokeniser?.getVocab() || [];
    }

    get loaded(): boolean {
        return !!this._model && !!this._tokeniser && !!this._config;
    }

    get config(): GPTConfig {
        if (!this._config) {
            throw new Error('Model configuration is not initialized.');
        }
        return this._config.gpt;
    }

    get model(): NanoGPT {
        if (!this._model) {
            throw new Error('Model is not initialized.');
        }
        return this._model;
    }

    get tokeniser(): ITokeniser {
        if (!this._tokeniser) {
            throw new Error('Tokeniser is not initialized.');
        }
        return this._tokeniser;
    }

    get status(): TeachableLLMStatus {
        return this._status;
    }

    get ready(): boolean {
        return this._status === 'ready' && !!this._model && !!this._tokeniser && this.tokeniser.trained;
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
            throw new Error('Model or tokeniser is not initialized.');
        }
        return saveModel(this._model, this._tokeniser, {
            ...options,
            name: options?.name || this.meta.name,
        });
    }

    static loadModel(data: Blob | Buffer | string): TeachableLLM {
        const teachableLLM = new TeachableLLM();
        loadModel(data)
            .then(({ model, tokeniser, name }) => {
                teachableLLM._model = model;
                teachableLLM._tokeniser = tokeniser;
                teachableLLM._config = model.config;
                if (name) {
                    teachableLLM.meta.name = name;
                }
                teachableLLM.setStatus('warmup');

                dummyPassTrainAsync(model)
                    .then((memoryReqs) => {
                        teachableLLM._memoryRequirements = memoryReqs;
                        teachableLLM.setStatus('ready');
                        teachableLLM.ee.emit('loaded');
                    })
                    .catch((err) => {
                        teachableLLM.setStatus('error');
                        teachableLLM.ee.emit('error', err);
                    });
            })
            .catch((err) => {
                teachableLLM.setStatus('error');
                teachableLLM.ee.emit('error', err);
            });

        return teachableLLM;
    }

    static create(tokeniserType: 'char' | 'bpe', config: Partial<GPTConfig> = {}) {
        const fullConfig = { ...defaultConfig, ...config };
        const tokeniser =
            tokeniserType === 'char' ? new CharTokeniser(fullConfig.vocabSize) : new BPETokeniser(fullConfig.vocabSize);
        const model = new NanoGPT(fullConfig);
        const tmodel = new TeachableLLM(tokeniser, model);
        tmodel.setStatus('warmup');

        dummyPassTrainAsync(model)
            .then((memoryReqs) => {
                tmodel._memoryRequirements = memoryReqs;

                if (tmodel.tokeniser.trained) {
                    tmodel.setStatus('ready');
                    tmodel.ee.emit('loaded');
                } else {
                    tmodel.setStatus('awaitingTokens');
                    tmodel.ee.emit('loaded');
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
                throw new Error('Model is not initialized.');
            }
            if (!this._config.layerConfig.profiler) {
                this._config.layerConfig.profiler = new MemoryProfiler();
            }
        } else {
            if (this._config?.layerConfig.profiler) {
                this._config.layerConfig.profiler = undefined;
            }
        }
    }

    getNumParams(): number {
        if (!this._model) {
            throw new Error('Model is not initialized.');
        }
        return this._model.getNumParams();
    }

    trainer() {
        if (!this._model || !this._tokeniser) {
            throw new Error('Model or tokeniser is not initialized.');
        }
        const trainer = new Trainer(this._model, this._tokeniser);
        trainer.on('start', () => this.setStatus('training'));
        trainer.on('stop', () => this.setStatus('ready'));
        trainer.on('log', async (step: TrainingLogEntry, progress: TrainingProgress) => {
            const listeners = this.ee.listeners('trainStep');
            for (const listener of listeners) {
                // These listeners can be async, so we await them
                await listener(step, progress);
            }
        });
        return trainer;
    }

    train(text: string[], options?: ITrainerOptions): Promise<void> {
        return this.trainer().train(text, options);
    }

    async trainTokeniser(text: string[]): Promise<number> {
        if (!this._tokeniser) {
            throw new Error('tokeniser_not_initialized.');
        }
        const tokenCount = await this._tokeniser.train(text);
        if (this._status === 'awaitingTokens') {
            this.setStatus('ready');
        }
        return tokenCount;
    }

    generator(): Generator {
        if (!this._model || !this._tokeniser) {
            throw new Error('model_or_tokeniser_not_initialized.');
        }
        const generator = new Generator(this._model, this._tokeniser);
        generator.on('start', () => {
            if (this.status === 'ready') this.setStatus('busy');
        });
        generator.on('stop', () => {
            if (this.status === 'busy') this.setStatus('ready');
        });
        return generator;
    }

    generateText(prompt?: string, options?: IGenerateOptions): Promise<string> {
        return this.generator().generate(prompt, options);
    }

    dispose() {
        this._model?.dispose();
    }

    on(event: 'status', listener: (status: TeachableLLMStatus) => void): void;
    on(event: 'error', listener: (error: Error) => void): void;
    on(event: 'trainStep', listener: (step: TrainingLogEntry, progress: TrainingProgress) => void): void;
    on(event: 'loaded', listener: () => void): void;
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
    off(event: 'error', listener: (error: Error) => void): void;
    off(event: 'trainStep', listener: (step: TrainingLogEntry, progress: TrainingProgress) => void): void;
    off(event: 'loaded', listener: () => void): void;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    off(event: TeachableLLMEvents, listener: (...args: any[]) => void): void {
        this.ee.off(event, listener);
    }
}
