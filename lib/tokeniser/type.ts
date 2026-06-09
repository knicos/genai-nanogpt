import type EE from 'eventemitter3';

export type Roles = 'user' | 'assistant' | 'system' | 'text';

export interface Conversation {
    role: Roles;
    content: string;
}

export interface ITokeniser extends EE<'trainStatus'> {
    id: string;
    datasetID?: string;
    train(text: Conversation[][], cb?: (vocab: number) => void, datasetID?: string): Promise<number>;
    //tokenise(text: string[], numeric?: boolean): Promise<string[][] | number[][]>;
    //detokenise(tokens: (number[] | Uint16Array)[]): Promise<string[]>;
    getVocab(): string[];
    getMerges(): [string, string][];
    destroy(): void;
    encode(text: string): number[];
    encodeConversation(conversation: Conversation[], completion?: boolean): number[];
    encodeConversation(
        conversation: Conversation[],
        completion: boolean,
        masking: boolean
    ): { tokens: number[]; mask: boolean[] };
    encodeConversation(
        conversation: Conversation[],
        completion?: boolean,
        masking?: boolean
    ): number[] | { tokens: number[]; mask: boolean[] };
    encodeSequence(text: string): number[];
    encodeAsSequence(conversation: Conversation[], completion?: boolean): number[];
    decode(tokens: number[] | Uint16Array): string;
    decodeConversation(tokens: number[] | Uint16Array): Conversation[];
    vocabSize: number;
    eosToken: number;
    bosToken: number;
    trained: boolean;
    getSpecialTokenIndex(token: string): number | undefined;
    isSpecialToken(index: number): boolean;
}
