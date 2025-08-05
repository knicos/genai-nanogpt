import type TF from '@tensorflow/tfjs';
import { defaultConfig, GPTConfig } from './config';
import { ITokeniser } from './tokeniser/type';
import CharTokeniser from './tokeniser/CharTokeniser';
import NanoGPT from './NanoGPTModel';
import { saveModel } from './utilities/save';
import { loadModel } from './utilities/load';

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

    static create(tf: typeof TF, config: Partial<GPTConfig> = {}) {
        const fullConfig = { ...defaultConfig, ...config };
        const model = new NanoGPT(tf, fullConfig);
        const tokeniser = new CharTokeniser();
        return new TeachableLLM(tf, tokeniser, model);
    }

    getNumParams(): number {
        return this.model.getNumParams();
    }

    // Train

    // Generate
}
