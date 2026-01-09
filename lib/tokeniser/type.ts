import type EE from 'eventemitter3';

export type Roles = 'user' | 'assistant' | 'system';

export interface Conversation {
    role: Roles;
    content: string;
}

export interface ITokeniser extends EE<'trainStatus'> {
    train(text: string[]): Promise<number>;
    tokenise(text: string[], numeric?: boolean): Promise<string[][] | number[][]>;
    detokenise(tokens: string[][] | number[][]): Promise<string[]>;
    getVocab(): string[];
    getMerges(): Promise<[string, string][]>;
    destroy(): void;
    encode(text: string): Promise<number[]>;
    encodeConversation(conversation: Conversation[]): Promise<number[]>;
    decode(tokens: number[]): Promise<string>;
    decodeConversation(tokens: number[]): Promise<Conversation[]>;
    vocabSize: number;
    eosToken: number;
    trained: boolean;
}
