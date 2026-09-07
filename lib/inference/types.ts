import type { Conversation } from '../tokeniser/type';
import { Tensor } from '@tensorflow/tfjs-core';

export interface IGeneratorOutput {
    outputTensor: Tensor;
    token: number;
    text: string;
    confidence: number | null;
    score: number | null;
    logits: number[] | null;
    scores: number[] | null;
    hiddenStates: number[][] | null;
    attention: number[][][][] | null;
    loss: number | null;
    multinomialRand: number | null;
    terminated: boolean;
}

export interface GeneratorConversation extends Conversation {
    _completed?: boolean;
    _timestamp?: number;
    _output?: IGeneratorOutput[];
}

export interface IGenerateOptions {
    // model?: string;
    input?: Conversation[] | string; /** Optional prompt. */
    previous_response_id?: string;
    temperature?: number;
    topK?: number; /** Select the top K best tokens. */
    topP?: number; /** Select the top P best tokens (nucleus sampling). */
    usePadding?: boolean;
    outputAttention?: boolean; /** All attention weights. */
    outputScores?: boolean; /** Softmax scores of the final logits (all tokens). */
    outputConfidence?: boolean; /** Entropy-based confidence in [0, 1] from the softmax distribution. */
    outputScore?: boolean; /** Selected token score (probability) */
    outputLogits?: boolean; /** Final output logits. */
    outputHiddenStates?: 'embedding' | 'logits' | 'softmax' | 'all';
    outputLoss?: boolean;
    outputMultinomialRand?: boolean;
    targets?: number[]; /** Optional target tokens for loss calculation. */
    loraName?: string; /** Optional LoRA name to use for inference. */
    maxLength?: number; /** Maximum length of the generated text. */
    noCache?: boolean; /** Do not use key/value cache for attention. */
    allowSpecial?: boolean; /** Keep special tokens in the output. */
    nonConversational?: boolean; /** Do not use turn taking tokens. */
    continuation?: boolean; /** Continue the previous response without adding a new turn. */
    chunkSize?: number; /** Number of tokens to generate before returning a partial response. */
    background?: boolean; /** Run the generation in the background and return immediately. */
    _onChunk?: (output: IGeneratorOutput) => void | Promise<void>;
}

export interface IGeneratorResponse {
    output: GeneratorConversation[] | null;
    id: string;
    done: boolean;
}

export interface BeamerOptions extends IGenerateOptions {
    maxBeamLength: number;
    beams: number;
    endOnWhiteSpace?: boolean;
}

export interface IBeam {
    tokens: number[];
    score: number;
    text: string;
}
