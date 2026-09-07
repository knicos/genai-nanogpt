import type { Conversation, ITokeniser } from '../tokeniser/type';
import EE from 'eventemitter3';
import { KVCache } from '../layers/CausalSelfAttention';
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
import CharTokeniser from '../tokeniser/CharTokeniser';
import multinomialCPU from '../utilities/multinomialCPU';
import Model, { ModelForwardAttributes } from '../models/model';
import topP from '../utilities/topP';
import { sparseSoftmaxCrossEntropy } from '../training/sparseCrossEntropy';
import { IGenerateOptions, GeneratorConversation, IGeneratorOutput } from './types';
import tokenisePrompt from './tokenisePrompt';
import { CHARS, getTokenConfidence, padArray } from './utilities';

interface JobItem {
    prompt?: Conversation[];
    options?: IGenerateOptions;
    resolve: (value: Conversation[] | PromiseLike<Conversation[]>) => void;
    reject: (reason?: unknown) => void;
}

export function isConversation(data: unknown): data is Conversation[] {
    return Array.isArray(data);
}

export interface IGenerator extends EE<'start' | 'stop' | 'tokens' | 'reset'> {
    generate(prompt: Conversation[], options?: IGenerateOptions): Promise<GeneratorConversation[]>;
    generate(options?: IGenerateOptions): Promise<GeneratorConversation[]>;
    step(prompt: Conversation[], options?: IGenerateOptions): Promise<GeneratorConversation[]>;
    step(options?: IGenerateOptions): Promise<GeneratorConversation[]>;
    stop(): void;
    getConversation(): GeneratorConversation[];
    getRawOutput(): IGeneratorOutput[];
    dispose(): void;
    reset(): void;
}

/**
 * Text generator using a NanoGPT model and a tokeniser.
 * This uses the forward method of the model to generate text token by token, including options for temperature, top-k, and top-p sampling.
 */
export default class Generator extends EE<'start' | 'stop' | 'tokens' | 'reset'> implements IGenerator {
    private active = false;
    private cache: KVCache[] | null = null;
    private initialPrompt: string | Conversation[] | null = null;
    private outputConversation: GeneratorConversation[] = [];
    private actualTokeniser: ITokeniser;
    private lastToken = -1;
    private lastLoss: number | null = null;
    private rawOutput: IGeneratorOutput[] = [];
    private jobQueue: JobItem[] = [];
    private processingJob = false;
    private startTime: number | null = null;
    private tokenCount = 0;

    constructor(
        private readonly model: Model<ModelForwardAttributes>,
        private readonly tokeniser: ITokeniser
    ) {
        super();
        this.actualTokeniser = tokeniser;
    }

    private shouldTerminate(allowSpecial: boolean, token: number): boolean {
        if (allowSpecial) return false;
        const assistantEndToken = this.tokeniser.getSpecialTokenIndex('<|assistant_end|>');
        if (token === this.actualTokeniser.eosToken || token === assistantEndToken) {
            return true;
        }
        return false;
    }

    /** Generate logits and select a token. */
    private async _generateToken(
        idx: Tensor,
        cache?: KVCache[],
        options?: IGenerateOptions
    ): Promise<IGeneratorOutput> {
        const temperature = options?.temperature ?? 1.0;
        const tK = options?.topK;
        const tP = options?.topP;
        const usePadding = options?.usePadding ?? false;

        const attrs: ModelForwardAttributes = {
            training: false,
            attentionScores: options?.outputAttention
                ? {
                      attentionOut: [],
                  }
                : undefined,
            cache,
            outputEmbeddings: !!options?.outputHiddenStates,
        };

        const [logits, loss] = tidy(() => {
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

            const logits = this.model.forward(attrs, padIdx);

            // Focus only on the last time step
            const lastTimeStep = logits.shape[1]! - 1 - padding;
            const lastLogits = logits.slice([0, lastTimeStep, 0], [logits.shape[0], 1, logits.shape[2]!]); // (b, 1, vocab_size)

            let lossValue: Tensor | undefined = undefined;
            if (options?.targets) {
                // Compute loss for the last time step
                const currentTarget = options.targets.shift();
                if (currentTarget !== undefined) {
                    const targetTensor = tensor2d([[currentTarget]], [1, 1], 'int32');
                    const loss = sparseSoftmaxCrossEntropy(lastLogits, targetTensor);
                    lossValue = loss.mean();
                    targetTensor.dispose();
                    loss.dispose();
                }
            }

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

            return [scaledLogits.squeeze([1]) as Tensor2D, lossValue];
        });

        let nextToken: Tensor;
        let probabilities: number[][] | undefined;
        let embeddings: { name: string; tensor: number[][] }[] | undefined;

        const rand = Math.random();

        if (tP) {
            // Top-p (nucleus) sampling
            const probs = softmax(logits);
            const probsArray = (await probs.array()) as number[][];
            probs.dispose();

            // Do topP on CPU.
            const renormProbs = topP(probsArray, tP);

            if (options?.outputScores || options?.outputConfidence) {
                probabilities = probsArray;
            }

            // Do the multinomial on the CPU
            nextToken = multinomialCPU(renormProbs, rand);
        } else if (tK) {
            const { values: topKValues, indices: topKIndices } = topk(logits, tK);
            // FIXME: Broken in Tensorflow.js for WebGPU backend
            //console.warn('Using broken multinomial');
            const sampledIdx = multinomial(topKValues, 1);
            nextToken = gather(topKIndices, sampledIdx, 1);

            topKValues.dispose();
            topKIndices.dispose();
            sampledIdx.dispose();
        } else {
            // FIXME: Broken in Tensorflow.js for WebGPU backend
            //console.warn('Using broken multinomial');
            nextToken = multinomial(logits, 1);
            if (options?.outputScores || options?.outputConfidence) {
                const probs = softmax(logits);
                probabilities = (await probs.array()) as number[][];
                probs.dispose();
            }
        }

        if (attrs.embeddings) {
            const filtered =
                options?.outputHiddenStates === 'all'
                    ? attrs.embeddings
                    : attrs.embeddings.filter((e) => e.name.startsWith('block_output_'));
            const promises = filtered.map(async (e) => {
                const seqLen = e.tensor.shape[1]!;
                const lastStep = e.tensor.slice([0, seqLen - 1, 0], [e.tensor.shape[0], 1, e.tensor.shape[2]!]);
                e.tensor.dispose();
                const squeezed = lastStep.squeeze([1]);
                lastStep.dispose();

                if (options?.outputHiddenStates === 'softmax') {
                    const projected = this.model.project(squeezed);
                    squeezed.dispose();
                    const softmaxed = softmax(projected, -1);
                    projected.dispose();
                    const result = { name: e.name, tensor: (await softmaxed.array()) as number[][] };
                    softmaxed.dispose();
                    return result;
                } else if (options?.outputHiddenStates === 'logits') {
                    const projected = this.model.project(squeezed);
                    squeezed.dispose();
                    const result = { name: e.name, tensor: (await projected.array()) as number[][] };
                    projected.dispose();
                    return result;
                } else {
                    const arr = (await squeezed.array()) as number[][];
                    squeezed.dispose();
                    return { name: e.name, tensor: arr };
                }
            });
            const embeddingsResult = await Promise.all(promises);
            embeddings = embeddingsResult;
        }

        const reshaped = nextToken.reshape([1, 1]);
        nextToken.dispose();
        nextToken = reshaped;

        const tokenNumber = ((await nextToken.array()) as number[][])[0][0];
        const tokenText = this.actualTokeniser.decode([tokenNumber]);
        this.lastToken = tokenNumber;
        const terminated = this.shouldTerminate(options?.allowSpecial ?? false, tokenNumber);

        const output: IGeneratorOutput = {
            outputTensor: nextToken,
            token: tokenNumber,
            text: tokenText,
            confidence: options?.outputConfidence && probabilities ? getTokenConfidence(probabilities[0]) : null,
            score: options?.outputScore && probabilities ? probabilities[0][tokenNumber] : null,
            logits: options?.outputLogits ? ((await logits.array()) as number[][])[0] : null,
            scores: options?.outputScores && probabilities ? probabilities[0] : null,
            hiddenStates: embeddings ? embeddings.map((e) => e.tensor[0]) : null,
            attention: options?.outputAttention
                ? ((await Promise.all(
                      attrs.attentionScores?.attentionOut?.map((a) => a.array()) ?? []
                  )) as number[][][][])
                : null,
            loss: this.lastLoss,
            multinomialRand: rand,
            terminated,
        };

        logits.dispose();

        if (loss) {
            const lossValue = (await loss.array()) as number;
            loss.dispose();
            output.loss = lossValue;
        }

        this.rawOutput.push(output);
        if (!options?.chunkSize || this.tokenCount++ % options.chunkSize === 0) {
            this.emit('tokens', output);
            if (options?._onChunk) {
                await options._onChunk(output);
            }
        }
        return output;
    }

    /** Generate multiple tokens in a loop and produce text */
    private async _generate(options?: IGenerateOptions, hasPrompt?: boolean): Promise<GeneratorConversation[]> {
        let appended = false;

        // Begin a new assistant response in conversation
        if (
            //this.lastToken < 0 ||
            this.outputConversation.length === 0 ||
            this.outputConversation[this.outputConversation.length - 1]._completed ||
            (this.outputConversation[this.outputConversation.length - 1].role !== 'assistant' &&
                options?.nonConversational !== true) ||
            (this.outputConversation[this.outputConversation.length - 1].role !== 'text' &&
                options?.nonConversational === true)
        ) {
            this.outputConversation.push({
                role: options?.nonConversational === true ? 'text' : 'assistant',
                content: '',
                _timestamp: Date.now(),
            });
            appended = true;
            this.resetCache(!options?.noCache);
        } else if (this.lastToken < 0 || hasPrompt) {
            this.resetCache(!options?.noCache);
        }

        let inputTensor =
            this.lastToken >= 0 && this.cache
                ? tensor2d([this.lastToken], [1, 1], 'int32')
                : await tokenisePrompt(
                      this.actualTokeniser,
                      this.model.config.blockSize,
                      hasPrompt
                          ? appended
                              ? this.outputConversation.slice(0, -1)
                              : this.outputConversation
                          : undefined,
                      options
                  );

        const maxTokens = options?.maxLength ?? 1000;

        // Loop in the model to generate text until eos or max length
        for (let i = 0; i < maxTokens; i++) {
            if (!this.active) {
                break;
            }

            const output = await this._generateToken(inputTensor, this.cache ? this.cache : undefined, {
                ...options,
                usePadding: !this.cache,
            });

            if (this.cache) {
                inputTensor.dispose();
                inputTensor = output.outputTensor;
            } else {
                const oldInput = inputTensor;
                inputTensor = concat([inputTensor, output.outputTensor], 1);
                oldInput.dispose();
            }

            const currentConversation = this.outputConversation[this.outputConversation.length - 1];

            if (!this.cache) {
                output.outputTensor.dispose();
            }
            if (output.terminated) {
                currentConversation._completed = true;
                break;
            }
            if (i === maxTokens - 1 && maxTokens > 1) {
                currentConversation._completed = true;
                output.terminated = true;
            }

            currentConversation.content += output.text;
            if (!currentConversation._output) {
                currentConversation._output = [];
            }
            currentConversation._output.push(output);
        }

        inputTensor.dispose();
        return this.outputConversation;
    }

    private resetCache(remake?: boolean) {
        if (this.cache) {
            this.cache.forEach((c) => {
                if (c) {
                    if (c.k) c.k.dispose();
                    if (c.v) c.v.dispose();
                    c.k = undefined;
                    c.v = undefined;
                    c.cumulativeLength = 0;
                    c.length = 0;
                }
            });
            if (!remake) {
                this.cache = null;
            }
        }
        this.lastToken = -1;
    }

    public reset() {
        this.resetCache();
        this.outputConversation = [];
        this.initialPrompt = null;
        this.rawOutput = [];
        this.lastLoss = null;
        this.emit('reset');
    }

    public dispose() {
        this.reset();
    }

    private initialise(prompt?: Conversation[], options?: IGenerateOptions) {
        if (this.cache && options?.noCache) {
            this.reset();
        }

        this.initialPrompt = prompt || null;
        if (this.lastToken === -1) {
            this.outputConversation = (this.initialPrompt || []).slice();
        } else if (prompt && prompt.length > this.outputConversation.length) {
            // If the conversation is now longer, update it
            this.outputConversation = (this.initialPrompt || []).slice();
            this.resetCache();
        }

        if (
            !this.cache &&
            !options?.noCache &&
            (this.model.config.modelType !== 'GenAI_NanoGPT_v1' || this.model.config.useRope)
        ) {
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

        if (options?.loraName) {
            this.model.attachLoRA(options.loraName);
        } else if (this.model.hasLoRA()) {
            this.model.detachLoRA();
        }
    }

    async step(prompt: Conversation[], options?: IGenerateOptions): Promise<GeneratorConversation[]>;
    async step(options?: IGenerateOptions): Promise<GeneratorConversation[]>;
    public async step(
        promptOrOptions?: Conversation[] | IGenerateOptions,
        options?: IGenerateOptions
    ): Promise<GeneratorConversation[]> {
        const stepOptions = { ...options, maxLength: 1 };
        if (isConversation(promptOrOptions)) {
            return this.generate(promptOrOptions, stepOptions);
        } else {
            return this.generate({ ...promptOrOptions, ...stepOptions });
        }
    }

    async generate(prompt: Conversation[], options?: IGenerateOptions): Promise<GeneratorConversation[]>;
    async generate(options?: IGenerateOptions): Promise<GeneratorConversation[]>;
    public async generate(
        promptOrOptions?: Conversation[] | IGenerateOptions,
        options?: IGenerateOptions
    ): Promise<GeneratorConversation[]> {
        let prompt: Conversation[] | undefined = undefined;
        if (Array.isArray(promptOrOptions)) {
            prompt = promptOrOptions;
        } else if (typeof promptOrOptions === 'object') {
            options = promptOrOptions;
        }

        if (this.processingJob) {
            if (this.jobQueue.length > 10) {
                throw new Error('Job queue is too long, rejecting new job');
            }
            return new Promise<GeneratorConversation[]>((resolve, reject) => {
                this.jobQueue.push({ prompt, options, resolve, reject });
            });
        }

        this.processingJob = true;
        this.startTime = Date.now();
        try {
            const result = await this.startJob(prompt, options);
            this.processingJob = false;

            // Process next job in the queue
            if (this.jobQueue.length > 0) {
                const nextJob = this.jobQueue.shift()!;
                this.generate(nextJob.prompt || [], nextJob.options)
                    .then(nextJob.resolve)
                    .catch(nextJob.reject);
            }

            return result;
        } catch (error) {
            this.processingJob = false;
            throw error;
        }
    }

    private async startJob(prompt?: Conversation[], options?: IGenerateOptions) {
        this.initialise(prompt, options);
        this.active = true;

        this.model.metaData.generationSettings = options;

        if (options?.maxLength !== 1) this.emit('start');
        const result = this._generate(options, !!prompt);
        const r = await result;
        this.active = false;

        if (this.startTime !== null) {
            //const endTime = Date.now();
            //const duration = endTime - this.startTime;
            this.startTime = null;
            /*this.model.metaData.actionLog = this.model.metaData.actionLog || [];
            this.model.metaData.actionLog.push({
                action: 'generate',
                timestamp: endTime,
                duration,
                tokensProcessed: this.rawOutput.length,
                options: options || {},
            });*/
        }

        this.emit('stop');
        return r;
    }

    public getQueueLength() {
        return this.jobQueue.length;
    }

    public stop() {
        this.active = false;
    }

    public getConversation(): Conversation[] {
        return this.outputConversation;
    }

    public getRawOutput(): IGeneratorOutput[] {
        return this.rawOutput;
    }
}
