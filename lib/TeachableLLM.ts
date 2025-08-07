import type TF from '@tensorflow/tfjs';
import { defaultConfig, GPTConfig } from './config';
import { ITokeniser } from './tokeniser/type';
import NanoGPT from './NanoGPTModel';
import { saveModel } from './utilities/save';
import { loadModel } from './utilities/load';
import Generator, { IGenerateOptions } from './Generator';
import Trainer, { ITrainerOptions } from './Trainer';
import EE from 'eventemitter3';
import { dummyPassAsync } from './utilities/dummy';

type TeachableLLMStatus = 'warmup' | 'ready' | 'training' | 'loading' | 'busy' | 'error';

export default class TeachableLLM extends EE<'status' | 'error'> {
    public readonly config: GPTConfig;
    public readonly model: NanoGPT;
    public readonly tf: typeof TF;
    public readonly tokeniser: ITokeniser;
    private _status: TeachableLLMStatus = 'loading';

    constructor(tf: typeof TF, tokeniser: ITokeniser, model: NanoGPT) {
        super();
        this.tf = tf;
        this.config = model.config;
        this.tokeniser = tokeniser;
        this.model = model;
    }

    get status(): TeachableLLMStatus {
        return this._status;
    }

    private setStatus(status: TeachableLLMStatus) {
        if (this._status !== status) {
            this._status = status;
            this.emit('status', status);
        }
    }

    saveModel(): Promise<Blob> {
        return saveModel(this.model, this.tokeniser);
    }

    static async loadModel(tf: typeof TF, data: Blob | Buffer | string): Promise<TeachableLLM> {
        const { model, tokeniser } = await loadModel(tf, data);
        const teachableLLM = new TeachableLLM(tf, tokeniser, model);
        teachableLLM.setStatus('warmup');
        dummyPassAsync(model)
            .then(() => {
                teachableLLM.setStatus('ready');
            })
            .catch((err) => {
                teachableLLM.setStatus('error');
                teachableLLM.emit('error', err);
            });
        return teachableLLM;
    }

    static create(tf: typeof TF, tokeniser: ITokeniser, config: Partial<GPTConfig> = {}) {
        const fullConfig = { ...defaultConfig, ...config };
        fullConfig.vocabSize = tokeniser.vocabSize;
        const model = new NanoGPT(tf, fullConfig);
        return new TeachableLLM(tf, tokeniser, model);
    }

    getNumParams(): number {
        return this.model.getNumParams();
    }

    trainer() {
        const trainer = new Trainer(this.model, this.tokeniser);
        trainer.on('start', () => this.setStatus('training'));
        trainer.on('stop', () => this.setStatus('ready'));
        return trainer;
    }

    train(text: string[], options?: ITrainerOptions): Promise<void> {
        return this.trainer().train(text, options);
    }

    generator(): Generator {
        const generator = new Generator(this.model, this.tokeniser);
        generator.on('start', () => this.setStatus('busy'));
        generator.on('stop', () => this.setStatus('ready'));
        return generator;
    }

    generateText(prompt?: string, options?: IGenerateOptions): Promise<string> {
        return this.generator().generate(prompt, options);
    }
}
