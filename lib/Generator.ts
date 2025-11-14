import NanoGPT, { GenerateOptions } from './NanoGPTModel';
import type { ITokeniser } from './tokeniser/type';
import EE from 'eventemitter3';
import { KVCache } from './layers/CausalSelfAttention';
import { concat, Tensor, tensor2d } from '@tensorflow/tfjs-core';
import { CharTokeniser } from './main';

const CHARS = [
    ...Array.from({ length: 95 }, (_, i) => String.fromCharCode(i + 32)), // ASCII
    // Spanish accented letters and punctuation
    ...'áéíóúüñ¿¡',
    // Finnish accented letters
    ...'äöÄÖÅå',
    // Greek letters
    ...'αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ',
    // Cyrillic letters
    ...'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',
];

function padArray(arr: string[], length: number): string[] {
    if (arr.length === length) return arr;
    if (arr.length > length) return arr.slice(0, length);
    return arr.concat(Array(length - arr.length).fill(''));
}

export interface IGenerateOptions extends GenerateOptions {
    maxLength?: number; /// Maximum length of the generated text
    noCache?: boolean;
}

export default class Generator extends EE<'start' | 'stop' | 'tokens'> {
    private active = false;
    private cache: KVCache[] | null = null;
    private initialPrompt: string | null = null;
    private outputText = '';
    private actualTokeniser: ITokeniser;
    private lastToken = -1;

    constructor(private readonly model: NanoGPT, private readonly tokeniser: ITokeniser) {
        super();
        this.actualTokeniser = tokeniser;
    }

    private async tokenisePrompt(tokeniser: ITokeniser, prompt?: string): Promise<Tensor> {
        const tokenisedPrompt = prompt ? await tokeniser.tokenise([prompt], true) : [[tokeniser.eosToken]];
        const inputTensor: Tensor = tensor2d(tokenisedPrompt, [1, tokenisedPrompt[0].length], 'int32');
        return inputTensor;
    }

    private async processResponse(
        tokeniser: ITokeniser,
        generatedToken: Tensor,
        attention: Tensor[] | undefined,
        probabilities: Tensor | undefined
    ): Promise<string | null> {
        const newToken = ((await generatedToken.array()) as number[][])[0][0];
        this.lastToken = newToken;
        if (newToken === this.tokeniser.eosToken) {
            return null;
        }
        const newText = await tokeniser.decode([newToken]);

        let attentionArray: number[][][] | undefined;
        if (attention) {
            attentionArray = await Promise.all(attention.map((a) => a.array().then((arr) => arr as number[][])));
            attention.forEach((a) => a.dispose());
        }

        let probabilitiesArray: number[][] | undefined;
        if (probabilities) {
            probabilitiesArray = (await probabilities.array()) as number[][];
            probabilities.dispose();
        }

        this.emit('tokens', [newToken], newText, attentionArray, probabilitiesArray);
        return newText;
    }

    private async _generate(options?: IGenerateOptions): Promise<string> {
        let inputTensor =
            this.lastToken >= 0 && this.cache
                ? tensor2d([this.lastToken], [1, 1], 'int32')
                : await this.tokenisePrompt(this.actualTokeniser, this.outputText);

        const maxTokens = options?.maxLength ?? 1000;

        // Loop in the model to generate text until eos or max length
        for (let i = 0; i < maxTokens; i++) {
            if (!this.active) {
                break;
            }

            const {
                output: generatedToken,
                probabilities,
                attention,
            } = await this.model.generate(inputTensor, this.cache ? this.cache : undefined, {
                ...options,
                usePadding: !this.cache,
            });

            if (this.cache) {
                inputTensor.dispose();
                inputTensor = generatedToken;
            } else {
                const oldInput = inputTensor;
                inputTensor = concat([inputTensor, generatedToken], 1);
                oldInput.dispose();
            }

            const newText = await this.processResponse(this.actualTokeniser, generatedToken, attention, probabilities);
            if (!this.cache) {
                generatedToken.dispose();
            }
            if (newText === null) {
                break;
            }
            this.outputText += newText;
        }

        inputTensor.dispose();
        return this.outputText;
    }

    public reset() {
        if (this.cache) {
            this.cache.forEach((c) => {
                if (c) {
                    if (c.k) c.k.dispose();
                    if (c.v) c.v.dispose();
                }
            });
            this.cache = null;
        }
        this.outputText = '';
        this.initialPrompt = null;
        this.lastToken = -1;
    }

    public dispose() {
        this.reset();
    }

    private initialise(prompt?: string, options?: IGenerateOptions) {
        const slicePrompt =
            prompt && prompt.length > this.model.config.gpt.blockSize
                ? prompt.slice(-this.model.config.gpt.blockSize)
                : prompt ?? null;

        if (this.cache && options?.noCache) {
            this.reset();
        }

        this.initialPrompt = slicePrompt || null;
        if (this.lastToken === -1) {
            this.outputText = this.initialPrompt || '';
        }

        if (!this.cache && !options?.noCache && this.model.config.gpt.useRope) {
            const cache: KVCache[] = new Array(this.model.config.gpt.nLayer);
            for (let i = 0; i < this.model.config.gpt.nLayer; i++) {
                cache[i] = { k: undefined, v: undefined, length: 0, cumulativeLength: 0 };
            }
            this.cache = cache;
            this.lastToken = -1;
        }

        const tokeniser = this.tokeniser.trained
            ? this.tokeniser
            : new CharTokeniser(padArray(CHARS, this.tokeniser.vocabSize));
        this.actualTokeniser = tokeniser;
    }

    public async step(prompt?: string, options?: IGenerateOptions): Promise<string> {
        const stepOptions = { ...options, maxLength: 1 };
        return this.generate(prompt, stepOptions);
    }

    public async generate(prompt?: string, options?: IGenerateOptions): Promise<string> {
        this.initialise(prompt, options);
        this.active = true;
        this.emit('start');
        const result = this._generate(options);
        const r = await result;
        this.active = false;
        this.emit('stop');
        return r;
    }

    public stop() {
        this.active = false;
    }

    public getText(): string {
        return this.outputText;
    }
}
