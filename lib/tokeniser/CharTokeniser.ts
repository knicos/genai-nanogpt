import EE from 'eventemitter3';
import { ITokeniser } from './type';

const specialTokens = ['<eos>', '<unk>'];

export default class CharTokeniser extends EE<'trainStatus'> implements ITokeniser {
    public vocabSize: number = 0;
    public eosToken = 0;
    public unkToken = 0;
    public vocab: string[] = [];
    private cache: Map<string, number> = new Map();

    constructor(vocabSize: number);
    constructor(vocab: string[]);
    constructor(vocabSizeOrVocab: number | string[]) {
        super();
        if (Array.isArray(vocabSizeOrVocab)) {
            this.vocab = vocabSizeOrVocab;
            if (this.vocab.length > 0) {
                this.vocabSize = this.vocab.length;
                this.eosToken = this.vocab.indexOf('<eos>');
                this.unkToken = this.vocab.indexOf('<unk>');
                // Try a few common fallback tokens if <unk> is not found
                if (this.unkToken === -1) {
                    this.unkToken = this.vocab.indexOf('<pad>');
                }
                if (this.unkToken === -1) {
                    this.unkToken = this.vocab.indexOf('_');
                }
                if (this.unkToken === -1) {
                    this.unkToken = this.vocab.indexOf(' ');
                }
                if (this.unkToken === -1) {
                    this.unkToken = this.eosToken;
                }
                this.vocab.forEach((token, index) => {
                    this.cache.set(token, index);
                });
            } else {
                throw new Error('Vocab cannot be empty');
            }
        } else {
            this.vocabSize = vocabSizeOrVocab;
        }
    }

    public get trained(): boolean {
        return this.vocab.length === this.vocabSize;
    }

    public destroy() {}

    public async train(text: string[]): Promise<number> {
        const flatText = text.map((t) => t.split('')).flat();
        const charSet = new Set(flatText);
        const charArray = Array.from(charSet);
        const actualSize = this.vocabSize - specialTokens.length;

        if (charArray.length > actualSize) {
            // Remove least common characters if we exceed the vocab size
            const counts = new Map<string, number>();
            flatText.forEach((char) => {
                counts.set(char, (counts.get(char) || 0) + 1);
            });
            charArray.sort((a, b) => (counts.get(a) || 0) - (counts.get(b) || 0));
            charArray.splice(0, charArray.length - actualSize);
        } else if (charArray.length < actualSize) {
            // Pad with <pad> if we have fewer characters than vocab size
            while (charArray.length < actualSize) {
                charArray.push('<pad>');
            }
        }

        charArray.sort((a, b) => a.charCodeAt(0) - b.charCodeAt(0));
        this.vocab = [...charArray, ...specialTokens];
        this.eosToken = this.vocab.indexOf('<eos>');
        this.unkToken = this.vocab.indexOf('<unk>');
        this.vocabSize = this.vocab.length;
        this.cache.clear();
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
                return t.split('').map((char) => this.cache.get(char) ?? this.unkToken);
            } else {
                return t.split('').map((char) => {
                    const index = this.cache.get(char);
                    return index !== undefined ? this.vocab[index] : '<unk>';
                });
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
