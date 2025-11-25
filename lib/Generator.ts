import type { ITokeniser } from './tokeniser/type';
import EE from 'eventemitter3';
import { KVCache } from './layers/CausalSelfAttention';
import {
    concat,
    gather,
    keep,
    multinomial,
    pad,
    softmax,
    Tensor,
    Tensor2D,
    tensor2d,
    tidy,
    topk,
} from '@tensorflow/tfjs-core';
import { CharTokeniser } from './main';
import multinomialCPU from './utilities/multinomialCPU';
import Model, { ModelForwardAttributes } from './models/model';

export interface GenerateOptions {
    temperature?: number;
    topK?: number;
    topP?: number;
    usePadding?: boolean;
    attentionScores?: boolean;
    includeProbabilities?: boolean;
    embeddings?: boolean;
}

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

/**
 * Text generator using a NanoGPT model and a tokeniser.
 * This uses the forward method of the model to generate text token by token, including options for temperature, top-k, and top-p sampling.
 */
export default class Generator extends EE<'start' | 'stop' | 'tokens'> {
    private active = false;
    private cache: KVCache[] | null = null;
    private initialPrompt: string | null = null;
    private outputText = '';
    private actualTokeniser: ITokeniser;
    private lastToken = -1;
    private attentionData: number[][][][][] = [];
    private probabilitiesData: number[][][] = [];
    private embeddingsData: number[][][][] = [];
    private tokens: number[] = [];

    constructor(private readonly model: Model<ModelForwardAttributes>, private readonly tokeniser: ITokeniser) {
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

        if (attention) {
            const attentionArray = await Promise.all(
                attention.map((a) => a.array().then((arr) => arr as number[][][]))
            );
            attention.forEach((a) => a.dispose());
            this.attentionData.push(attentionArray);
        }

        if (probabilities) {
            const probabilitiesArray = (await probabilities.array()) as number[][];
            probabilities.dispose();
            this.probabilitiesData.push(probabilitiesArray);
        }

        this.tokens.push(newToken);

        this.emit('tokens', [newToken], newText);
        return newText;
    }

    /** Generate logits and select a token. */
    private async _generateToken(
        idx: Tensor,
        cache?: KVCache[],
        options?: GenerateOptions
    ): Promise<{ output: Tensor; probabilities?: Tensor; attention?: Tensor[] }> {
        const temperature = options?.temperature ?? 1.0;
        const tK = options?.topK;
        const tP = options?.topP;
        const usePadding = options?.usePadding ?? false;

        const attrs: ModelForwardAttributes = {
            training: false,
            attentionScores: options?.attentionScores
                ? {
                      attentionOut: [],
                  }
                : undefined,
            cache,
            outputEmbeddings: options?.embeddings ?? false,
        };

        const logits = tidy(() => {
            const currentIdx = idx;

            // Crop sequence if it exceeds block size
            const seqLen = currentIdx.shape[1]!;
            const cropIdx =
                seqLen <= this.model.config.blockSize
                    ? currentIdx
                    : currentIdx.slice(
                          [0, seqLen - this.model.config.blockSize],
                          [currentIdx.shape[0], this.model.config.blockSize]
                      );
            const padding = usePadding ? this.model.config.blockSize - cropIdx.shape[1]! : 0;
            // In some cases padding is faster
            const padIdx =
                padding > 0
                    ? pad(cropIdx, [
                          [0, 0],
                          [0, padding],
                      ])
                    : cropIdx;

            const [logits] = this.model.forward(attrs, padIdx);

            // Focus only on the last time step
            const lastTimeStep = logits.shape[1]! - 1 - padding;
            const lastLogits = logits.slice([0, lastTimeStep, 0], [logits.shape[0], 1, logits.shape[2]!]); // (b, 1, vocab_size)

            // Double check that attention output is only the last step
            if (attrs.attentionScores?.attentionOut) {
                attrs.attentionScores.attentionOut.forEach((a, i) => {
                    if (a.shape[1]! !== 1) {
                        attrs.attentionScores!.attentionOut![i] = keep(
                            a.slice([0, lastTimeStep, 0], [a.shape[0], 1, a.shape[2]!])
                        );
                        a.dispose();
                    }
                });
            }

            logits.dispose();

            const scaledLogits = lastLogits.div(temperature);

            return scaledLogits.squeeze([1]) as Tensor2D;
        });

        let nextToken: Tensor;

        if (tP) {
            // Top-p (nucleus) sampling
            const probs = softmax(logits);
            const probsArray = (await probs.array()) as number[][];

            probs.dispose();
            const sorted = probsArray[0].map((p, i) => ({ prob: p, index: i })).sort((a, b) => b.prob - a.prob);

            let cumulativeProb = 0;
            const masked = new Array<number>(sorted.length).fill(0);
            for (const item of sorted) {
                cumulativeProb += item.prob;
                masked[item.index] = item.prob;
                if (cumulativeProb >= tP) {
                    break;
                }
            }

            // Renormalize
            const sumMasked = masked.reduce((a, b) => a + b, 0);
            const renormProbs = masked.map((p) => p / sumMasked);

            // Do the multinomial on the CPU
            nextToken = multinomialCPU(renormProbs);
        } else if (tK) {
            const { values: topKValues, indices: topKIndices } = topk(logits, tK);
            const sampledIdx = multinomial(topKValues, 1);
            nextToken = gather(topKIndices, sampledIdx, 1);

            topKValues.dispose();
            topKIndices.dispose();
            sampledIdx.dispose();
        } else {
            nextToken = multinomial(logits, 1);
        }

        let probabilities: Tensor | undefined;
        if (options?.includeProbabilities) {
            probabilities = softmax(logits);
        }

        if (attrs.embeddings) {
            this.embeddingsData.push(
                await Promise.all(
                    attrs.embeddings.map(async (e) => {
                        const arr = (await e.array()) as number[][];
                        e.dispose();
                        return arr;
                    })
                )
            );
        }

        const reshaped = nextToken.reshape([1, 1]);
        nextToken.dispose();
        nextToken = reshaped;

        logits.dispose();

        return { output: nextToken, probabilities, attention: attrs.attentionScores?.attentionOut };
    }

    /** Generate multiple tokens in a loop and produce text */
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
            } = await this._generateToken(inputTensor, this.cache ? this.cache : undefined, {
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
        this.attentionData = [];
        this.probabilitiesData = [];
        this.tokens = [];
    }

    public dispose() {
        this.reset();
    }

    private initialise(prompt?: string, options?: IGenerateOptions) {
        const slicePrompt =
            prompt && prompt.length > this.model.config.blockSize
                ? prompt.slice(-this.model.config.blockSize)
                : prompt ?? null;

        if (this.cache && options?.noCache) {
            this.reset();
        }

        this.initialPrompt = slicePrompt || null;
        if (this.lastToken === -1) {
            this.outputText = this.initialPrompt || '';
        }

        if (!this.cache && !options?.noCache && this.model.config.useRope) {
            const cache: KVCache[] = new Array(this.model.config.nLayer);
            for (let i = 0; i < this.model.config.nLayer; i++) {
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

    public getAttentionData(): number[][][][][] {
        return this.attentionData;
    }

    public getProbabilitiesData(): number[][][] {
        return this.probabilitiesData;
    }

    public getEmbeddingsData(): number[][][][] {
        return this.embeddingsData;
    }

    public getTokens(): number[] {
        return this.tokens;
    }
}
