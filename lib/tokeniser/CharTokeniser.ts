import BaseTokeniser, { SPECIALS } from './BaseTokeniser';

const specialTokens = ['<eos>', '<unk>'];

export default class CharTokeniser extends BaseTokeniser {
    public vocabSize: number = 0;
    public eosToken = 0;
    public bosToken = 0;
    public unkToken = 0;
    public vocab: string[] = [];
    private cache: Map<string, number> = new Map();
    private _trained: boolean = false;

    constructor(vocabSize: number);
    constructor(vocab: string[]);
    constructor(vocabSizeOrVocab: number | string[]) {
        super();
        if (Array.isArray(vocabSizeOrVocab)) {
            this.vocab = vocabSizeOrVocab;
            if (this.vocab.length > 0) {
                this.vocabSize = this.vocab.length;

                SPECIALS.forEach((token) => {
                    const index = this.vocab.indexOf(token);
                    if (index !== -1) {
                        this.addSpecialToken(token, index);
                    }
                });

                this.eosToken = this.getSpecialTokenIndex('<eos>')!;
                this.bosToken = this.getSpecialTokenIndex('<bos>') ?? this.eosToken;
                this.unkToken = this.getSpecialTokenIndex('') ?? -1;

                // Try a few common fallback tokens if <unk> is not found
                if (this.unkToken === -1) {
                    this.unkToken = this.vocab.indexOf('<unk>');
                }
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

                // Replace <pad> tokens with ''
                this.vocab = this.vocab.map((token) => (token === '<pad>' ? '' : token));

                this.vocab.forEach((token, index) => {
                    this.cache.set(token, index);
                });
            } else {
                throw new Error('Vocab cannot be empty');
            }
            this._trained = true;
        } else {
            this.vocabSize = vocabSizeOrVocab;
            this.vocab = new Array<string>(this.vocabSize).fill('');
            this.addSpecialTokens();
            this.eosToken = this.getSpecialTokenIndex('<eos>')!;
            this.bosToken = this.getSpecialTokenIndex('<bos>') ?? this.eosToken;
            this.unkToken = this.getSpecialTokenIndex('')!;

            // Special tokens don't really need to be cached
            this.vocab.forEach((token, index) => {
                this.cache.set(token, index);
            });
            this.cache.set('', this.unkToken);
        }
    }

    addToken(token: string, index?: number): number {
        if (this.cache.has(token)) {
            return this.cache.get(token)!;
        }

        let tokenIndex: number;
        if (index !== undefined) {
            tokenIndex = index;
        } else {
            tokenIndex = this.vocab.indexOf('', this.unkToken + 1);
            if (tokenIndex === -1) {
                tokenIndex = this.vocabSize;
            }
        }

        if (tokenIndex >= this.vocabSize) {
            throw new Error('Vocab size exceeded');
        }

        this.vocab[tokenIndex] = token;
        this.cache.set(token, tokenIndex);
        return tokenIndex;
    }

    public get trained(): boolean {
        return this.vocab.length === this.vocabSize && this._trained;
    }

    public destroy() {}

    public async train(text: string[]): Promise<number> {
        const flatText = text.map((t) => t.split('')).flat();
        const charSet = new Set(flatText);
        const charArray = Array.from(charSet);
        const firstPadIndex = this.vocab.indexOf('', this.unkToken + 1);
        const actualSize = this.vocabSize - specialTokens.length;

        if (firstPadIndex === -1) {
            return this.vocabSize; // No space left to add new characters
        }

        this._trained = true;

        if (charArray.length > actualSize) {
            // Remove least common characters if we exceed the vocab size
            const counts = new Map<string, number>();
            flatText.forEach((char) => {
                counts.set(char, (counts.get(char) || 0) + 1);
            });
            charArray.sort((a, b) => (counts.get(a) || 0) - (counts.get(b) || 0));
            charArray.splice(0, charArray.length - actualSize);
        }

        // charArray.sort((a, b) => a.charCodeAt(0) - b.charCodeAt(0));

        // Now merge charArray into existing vocab by replacing <pad> tokens
        let padIndex = firstPadIndex;
        if (padIndex !== -1) {
            const existingTokens = new Set(this.vocab);
            for (const char of charArray) {
                if (!existingTokens.has(char)) {
                    this.vocab[padIndex] = char;
                    existingTokens.add(char);
                    padIndex = this.vocab.indexOf('', padIndex + 1);
                    if (padIndex === -1) {
                        break;
                    }
                }
            }
        }

        this.cache.clear();
        this.vocab.forEach((token, index) => {
            this.cache.set(token, index);
        });

        this.emit('trainStatus', 'trained');
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
                    return index !== undefined ? this.vocab[index] : '';
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
