import NanoGPT, { GenerateOptions } from './NanoGPTModel';
import type { ITokeniser } from './tokeniser/type';
import EE from 'eventemitter3';
import { KVCache } from './layers/CausalSelfAttention';
import { concat, Tensor, tensor2d } from '@tensorflow/tfjs-core';

export interface IGenerateOptions extends GenerateOptions {
    maxLength?: number; /// Maximum length of the generated text
    noCache?: boolean;
}

export default class Generator extends EE<'start' | 'stop' | 'tokens'> {
    private active = false;

    constructor(private readonly model: NanoGPT, private readonly tokeniser: ITokeniser) {
        super();
    }

    private async tokenisePrompt(prompt?: string): Promise<Tensor> {
        const tokenisedPrompt = prompt ? await this.tokeniser.tokenise([prompt], true) : [[this.tokeniser.eosToken]];
        const inputTensor: Tensor = tensor2d(tokenisedPrompt, [1, tokenisedPrompt[0].length], 'int32');
        return inputTensor;
    }

    private async generateNoCache(prompt?: string, options?: IGenerateOptions): Promise<string> {
        let inputTensor = await this.tokenisePrompt(prompt);
        let outputText = prompt || '';

        const maxTokens = options?.maxLength ?? 1000;

        // Loop in the model to generate text until eos or max length
        for (let i = 0; i < maxTokens; i++) {
            if (!this.active) {
                break;
            }
            const {
                output: generatedToken,
                attention,
                probabilities,
            } = this.model.generate(inputTensor, undefined, options);

            const oldInput = inputTensor;
            inputTensor = concat([inputTensor, generatedToken], 1);
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
        generatedToken: Tensor,
        attention: Tensor | undefined,
        probabilities: Tensor | undefined
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

        const cache: KVCache[] = new Array(this.model.config.gpt.nLayer).fill(undefined);

        const maxTokens = options?.maxLength ?? 1000;

        // Loop in the model to generate text until eos or max length
        for (let i = 0; i < maxTokens; i++) {
            if (!this.active) {
                break;
            }
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

        cache.forEach((c) => {
            if (c) {
                c.k.dispose();
                c.v.dispose();
            }
        });

        inputTensor.dispose();
        return outputText;
    }

    public async generate(prompt?: string, options?: IGenerateOptions): Promise<string> {
        const slicePrompt =
            prompt && prompt.length > this.model.config.gpt.blockSize
                ? prompt.slice(-this.model.config.gpt.blockSize)
                : prompt;
        this.active = true;
        this.emit('start');
        const result =
            this.model.config.gpt.useRope && !options?.noCache && !options?.includeAttention
                ? this.generateCache(slicePrompt, options)
                : this.generateNoCache(slicePrompt, options);
        const r = await result;
        this.active = false;
        this.emit('stop');
        return r;
    }

    public stop() {
        this.active = false;
    }
}
