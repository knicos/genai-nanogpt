import type TF from '@tensorflow/tfjs';
import { defaultConfig, GPTConfig } from './config';
import { ITokeniser } from './tokeniser/type';
import NanoGPT from './NanoGPTModel';
import { saveModel, SaveOptions } from './utilities/save';
import { loadModel } from './utilities/load';
import Generator, { IGenerateOptions } from './Generator';
import Trainer, { ITrainerOptions } from './Trainer';
import EE from 'eventemitter3';
import { dummyPassAsync } from './utilities/dummy';
import { CharTokeniser } from './main';

type TeachableLLMStatus = 'warmup' | 'awaitingTokens' | 'ready' | 'training' | 'loading' | 'busy' | 'error';

export default class TeachableLLM extends EE<'status' | 'error' | 'trainStep'> {
    private _config?: GPTConfig;
    private _model?: NanoGPT;
    public readonly tf: typeof TF;
    private _tokeniser?: ITokeniser;
    private _status: TeachableLLMStatus = 'loading';

    constructor(tf: typeof TF, tokeniser?: ITokeniser, model?: NanoGPT) {
        super();
        this.tf = tf;
        this._config = model?.config;
        this._tokeniser = tokeniser;
        this._model = model;
    }

    get config(): GPTConfig {
        if (!this._config) {
            throw new Error('Model configuration is not initialized.');
        }
        return this._config;
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

    private setStatus(status: TeachableLLMStatus) {
        if (this._status !== status) {
            this._status = status;
            this.emit('status', status);
        }
    }

    saveModel(options?: SaveOptions): Promise<Blob> {
        if (!this._model || !this._tokeniser) {
            throw new Error('Model or tokeniser is not initialized.');
        }
        return saveModel(this._model, this._tokeniser, options);
    }

    static loadModel(tf: typeof TF, data: Blob | Buffer | string): TeachableLLM {
        const teachableLLM = new TeachableLLM(tf);
        loadModel(tf, data)
            .then(({ model, tokeniser }) => {
                teachableLLM._model = model;
                teachableLLM._tokeniser = tokeniser;
                teachableLLM._config = model.config;
                teachableLLM.setStatus('warmup');
                dummyPassAsync(model)
                    .then(() => {
                        teachableLLM.setStatus('ready');
                    })
                    .catch((err) => {
                        teachableLLM.setStatus('error');
                        teachableLLM.emit('error', err);
                    });
            })
            .catch((err) => {
                teachableLLM.setStatus('error');
                teachableLLM.emit('error', err);
            });

        return teachableLLM;
    }

    static create(tf: typeof TF, config: Partial<GPTConfig> = {}) {
        const fullConfig = { ...defaultConfig, ...config };
        const tokeniser = new CharTokeniser(fullConfig.vocabSize);
        const model = new NanoGPT(tf, fullConfig);
        const tmodel = new TeachableLLM(tf, tokeniser, model);
        tmodel.setStatus('warmup');
        dummyPassAsync(model)
            .then(() => {
                tmodel.setStatus('awaitingTokens');
                tmodel.tokeniser.once('trainStatus', (status) => {
                    if (status === 'trained') {
                        tmodel.setStatus('ready');
                    }
                });
            })
            .catch((err) => {
                tmodel.setStatus('error');
                tmodel.emit('error', err);
            });
        return tmodel;
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
        trainer.on('log', async (step) => {
            const listeners = this.listeners('trainStep');
            for (const listener of listeners) {
                // These listeners can be async, so we await them
                await listener(step);
            }
        });
        return trainer;
    }

    train(text: string[], options?: ITrainerOptions): Promise<void> {
        return this.trainer().train(text, options);
    }

    generator(): Generator {
        if (!this._model || !this._tokeniser) {
            throw new Error('Model or tokeniser is not initialized.');
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
}
