import EE from 'eventemitter3';
import { ITokeniser } from './type';
import BPE from './bpe';

export default class NodeTokeniser extends EE<'trainStatus'> implements ITokeniser {
    public vocabSize: number = 0;
    public eosToken = 0;
    private bpe = new BPE();

    constructor(vocabSize: number);
    constructor(vocab: string[], merges: [string, string][]);
    constructor(vocabOrSize: string[] | number, merges?: [string, string][]) {
        super();
        if (Array.isArray(vocabOrSize)) {
            this.bpe = new BPE(vocabOrSize, merges);
            this.vocabSize = vocabOrSize.length;
        } else {
            this.vocabSize = vocabOrSize;
        }
    }

    public get trained(): boolean {
        return this.vocabSize > 0;
    }

    public destroy() {}

    public async train(text: string[]): Promise<number> {
        this.bpe.train(text, this.vocabSize);
        this.vocabSize = this.bpe.getVocab().length;
        return this.vocabSize;
    }

    public async tokenise(text: string[], numeric: true): Promise<number[][]>;
    public async tokenise(text: string[]): Promise<string[][]>;
    public async tokenise(text: string[], numeric?: boolean): Promise<string[][] | number[][]> {
        const tokens = numeric ? this.bpe.tokenise(text, true) : this.bpe.tokenise(text);
        return tokens;
    }

    public async detokenise(tokens: number[][]): Promise<string[]> {
        const vocab = this.bpe.getVocab();
        const text = tokens.map((t) => t.map((tt) => vocab[tt]).join(''));
        return text;
    }

    public async encode(text: string): Promise<number[]> {
        return (await this.tokenise([text], true))[0];
    }

    public async decode(tokens: number[]): Promise<string> {
        return (await this.detokenise([tokens]))[0];
    }

    public getVocab(): string[] {
        return this.bpe.getVocab();
    }

    public async getMerges(): Promise<[string, string][]> {
        return this.bpe.getMerges();
    }

    public async createTrainingData(text: string[], windowSize: number = 5): Promise<[number[], number[]]> {
        const tokenised = this.bpe.tokenise(text, true);
        const inputs: number[] = [];
        const targets: number[] = [];

        for (let i = 0; i < tokenised.length - windowSize; i++) {
            inputs.push(...tokenised[i].slice(0, windowSize));
            targets.push(tokenised[i + 1][0]); // Predict the next token
        }

        return [inputs, targets];
    }
}
