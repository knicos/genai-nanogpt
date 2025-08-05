import EE from 'eventemitter3';
import { ITokeniser } from './type';

export default class CharTokeniser extends EE<'trainStatus'> implements ITokeniser {
    public vocabSize: number = 0;
    public eosToken = 0;
    public vocab: string[] = [];
    private cache: Map<string, number> = new Map();

    constructor(vocab?: string[]) {
        super();
        this.vocab = vocab || [];
        if (this.vocab.length > 0) {
            this.vocabSize = this.vocab.length;
            this.eosToken = this.vocab.indexOf('<eos>');
            this.vocab.forEach((token, index) => {
                this.cache.set(token, index);
            });
        }
    }

    public get trained(): boolean {
        return this.vocabSize > 0;
    }

    public destroy() {}

    public async train(text: string[]): Promise<number> {
        const charSet = new Set(text.map((t) => t.split('')).flat());
        const charArray = Array.from(charSet);
        charArray.sort((a, b) => a.charCodeAt(0) - b.charCodeAt(0));
        this.vocab = [...charArray, '<eos>'];
        this.eosToken = this.vocab.indexOf('<eos>');
        this.vocabSize = this.vocab.length;
        this.vocab.forEach((token, index) => {
            this.cache.set(token, index);
        });
        return this.vocabSize;
    }

    public async tokenise(text: string[], numeric: true): Promise<number[][]>;
    public async tokenise(text: string[]): Promise<string[][]>;
    public async tokenise(text: string[], numeric?: boolean): Promise<string[][] | number[][]> {
        if (!this.trained) {
            throw new Error('Tokeniser not trained');
        }

        const tokenised: (string[] | number[])[] = text.map((t) => {
            if (numeric) {
                return t.split('').map((char) => this.cache.get(char) ?? -1);
            } else {
                return t.split('');
            }
        });

        return tokenised as string[][] | number[][];
    }

    public async detokenise(tokens: number[][]): Promise<string[]> {
        const text = tokens.map((t) => t.map((tt) => this.vocab[tt]).join(''));
        return text;
    }

    public async encode(text: string): Promise<number[]> {
        return (await this.tokenise([text], true))[0];
    }

    public async decode(tokens: number[]): Promise<string> {
        return (await this.detokenise([tokens]))[0];
    }

    public getVocab(): string[] {
        return this.vocab;
    }

    public async getMerges(): Promise<[string, string][]> {
        // Char tokeniser does not use merges
        return [];
    }

    public async createTrainingData(text: string[], windowSize: number = 5): Promise<[number[], number[]]> {
        const tokenised = await this.tokenise(text, true);
        const inputs: number[] = [];
        const targets: number[] = [];

        for (let i = 0; i < tokenised.length - windowSize; i++) {
            inputs.push(...tokenised[i].slice(0, windowSize));
            targets.push(tokenised[i + 1][0]); // Predict the next token
        }

        return [inputs, targets];
    }
}
