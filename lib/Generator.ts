import type TF from '@tensorflow/tfjs';
import NanoGPT, { GenerateOptions } from './NanoGPTModel';
import { ITokeniser } from './tokeniser/type';
import EE from 'eventemitter3';
import { KVCache } from './layers/CausalSelfAttention';

export interface IGenerateOptions extends GenerateOptions {
    maxLength?: number; /// Maximum length of the generated text
}

export default class Generator extends EE<'start' | 'stop' | 'tokens'> {
    constructor(private readonly model: NanoGPT, private readonly tokeniser: ITokeniser) {
        super();
    }

    private async tokenisePrompt(prompt?: string): Promise<TF.Tensor> {
        const tokenisedPrompt = prompt ? await this.tokeniser.tokenise([prompt], true) : [[this.tokeniser.eosToken]];
        const inputTensor: TF.Tensor = this.model.tf.tensor2d(tokenisedPrompt, [1, tokenisedPrompt[0].length], 'int32');
        return inputTensor;
    }

    private async generateNoCache(prompt?: string, options?: IGenerateOptions): Promise<string> {
        let inputTensor = await this.tokenisePrompt(prompt);
        let outputText = prompt || '';

        const maxTokens = options?.maxLength ?? 1000;

        // Loop in the model to generate text until eos or max length
        for (let i = 0; i < maxTokens; i++) {
            const {
                output: generatedToken,
                attention,
                probabilities,
            } = this.model.generate(inputTensor, undefined, options);

            const oldInput = inputTensor;
            inputTensor = this.model.tf.concat([inputTensor, generatedToken], 1);
            oldInput.dispose();

            const newText = await this.processResponse(generatedToken, attention, probabilities);
            generatedToken.dispose();

            if (newText === null) {
                break;
            }
            outputText += newText;
        }

        inputTensor.dispose();
        return outputText;
    }

    private async processResponse(
        generatedToken: TF.Tensor,
        attention: TF.Tensor | undefined,
        probabilities: TF.Tensor | undefined
    ): Promise<string | null> {
        const newToken = ((await generatedToken.array()) as number[][])[0][0];
        if (newToken === this.tokeniser.eosToken) {
            return null;
        }
        const newText = await this.tokeniser.decode([newToken]);

        let attentionArray: number[][] | undefined;
        if (attention) {
            attentionArray = (await attention.array()) as number[][];
            attention.dispose();
        }

        let probabilitiesArray: number[][] | undefined;
        if (probabilities) {
            probabilitiesArray = (await probabilities.array()) as number[][];
            probabilities.dispose();
        }

        this.emit('tokens', [newToken], newText, attentionArray, probabilitiesArray);
        return newText;
    }

    private async generateCache(prompt?: string, options?: IGenerateOptions): Promise<string> {
        let inputTensor = await this.tokenisePrompt(prompt);
        let outputText = prompt || '';

        const cache: KVCache[] = new Array(this.model.config.nLayer).fill(undefined);

        const maxTokens = options?.maxLength ?? 1000;

        // Loop in the model to generate text until eos or max length
        for (let i = 0; i < maxTokens; i++) {
            const {
                output: generatedToken,
                attention,
                probabilities,
            } = this.model.generate(inputTensor, cache, {
                ...options,
                usePadding: false,
            });

            inputTensor.dispose();
            inputTensor = generatedToken;

            const newText = await this.processResponse(generatedToken, attention, probabilities);
            if (newText === null) {
                break;
            }
            outputText += newText;
        }

        inputTensor.dispose();
        return outputText;
    }

    public async generate(prompt?: string, options?: IGenerateOptions): Promise<string> {
        this.emit('start');
        const result = this.model.config.useRope
            ? this.generateCache(prompt, options)
            : this.generateNoCache(prompt, options);
        this.emit('stop');
        return result;
    }
}
