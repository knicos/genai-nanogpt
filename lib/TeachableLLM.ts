import type TF from '@tensorflow/tfjs';
import { defaultConfig, GPTConfig } from './config';
import { ITokeniser } from './tokeniser/type';
import NanoGPT from './NanoGPTModel';
import { saveModel } from './utilities/save';
import { loadModel } from './utilities/load';
import Generator, { IGenerateOptions } from './Generator';
import Trainer, { ITrainerOptions } from './Trainer';

export default class TeachableLLM {
    public readonly config: GPTConfig;
    public readonly model: NanoGPT;
    public readonly tf: typeof TF;
    public readonly tokeniser: ITokeniser;

    constructor(tf: typeof TF, tokeniser: ITokeniser, model: NanoGPT) {
        this.tf = tf;
        this.config = model.config;
        this.tokeniser = tokeniser;
        this.model = model;
    }

    saveModel(): Promise<Blob> {
        return saveModel(this.model, this.tokeniser);
    }

    static async loadModel(tf: typeof TF, data: Blob | Buffer | string): Promise<TeachableLLM> {
        const { model, tokeniser } = await loadModel(tf, data);
        return new TeachableLLM(tf, tokeniser, model);
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
        return new Trainer(this.model, this.tokeniser);
    }

    train(text: string[], options?: ITrainerOptions): Promise<void> {
        return this.trainer().train(text, options);
    }

    generator(): Generator {
        return new Generator(this.model, this.tokeniser);
    }

    generateText(prompt?: string, options?: IGenerateOptions): Promise<string> {
        return this.generator().generate(prompt, options);
    }
}
