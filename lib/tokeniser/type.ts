import type EE from 'eventemitter3';

export type Roles = 'user' | 'assistant' | 'system';

export interface Conversation {
    role: Roles;
    content: string;
}

export interface ITokeniser extends EE<'trainStatus'> {
    train(text: string[]): Promise<number>;
    //tokenise(text: string[], numeric?: boolean): Promise<string[][] | number[][]>;
    //detokenise(tokens: (number[] | Uint16Array)[]): Promise<string[]>;
    getVocab(): string[];
    getMerges(): [string, string][];
    destroy(): void;
    encode(text: string): number[];
    encodeConversation(conversation: Conversation[], completion?: boolean): number[];
    encodeSequence(text: string): number[];
    decode(tokens: number[] | Uint16Array): string;
    decodeConversation(tokens: number[] | Uint16Array): Conversation[];
    vocabSize: number;
    eosToken: number;
    bosToken: number;
    trained: boolean;
    getSpecialTokenIndex(token: string): number | undefined;
    isSpecialToken(index: number): boolean;
}
