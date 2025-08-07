import type EE from 'eventemitter3';

export interface ITokeniser extends EE<'trainStatus'> {
    train(text: string[]): Promise<number>;
    tokenise(text: string[], numeric?: boolean): Promise<string[][] | number[][]>;
    detokenise(tokens: string[][] | number[][]): Promise<string[]>;
    getVocab(): string[];
    getMerges(): Promise<[string, string][]>;
    destroy(): void;
    encode(text: string): Promise<number[]>;
    decode(tokens: number[]): Promise<string>;
    vocabSize: number;
    eosToken: number;
    trained: boolean;
}
