import type { Conversation } from '../tokeniser/type';

export interface GeneratorConversation extends Conversation {
    _completed?: boolean;
    _timestamp?: number;
}

export interface GenerateOptions {
    temperature?: number;
    topK?: number;
    topP?: number;
    usePadding?: boolean;
    attentionScores?: boolean;
    includeProbabilities?: boolean;
    embeddings?: 'embedding' | 'logits' | 'softmax' | 'all';
    targets?: number[];
    loraName?: string;
}
