import parseTokens from '../utilities/tokenParse';

interface TokenPair {
    a: string;
    b: string;
    count: number;
    instances: Set<number>;
}

interface InternalTokenState {
    tokens: string[][];
    pairs: Map<string, TokenPair>;
}

function initPairs(tokens: string[][]): InternalTokenState {
    const pairs = new Map<string, TokenPair>();
    for (let j = 0; j < tokens.length; j++) {
        const tokenSet = tokens[j];
        for (let i = 0; i < tokenSet.length - 1; i++) {
            const pair = `${tokenSet[i]}${tokenSet[i + 1]}`;
            const entry = pairs.get(pair) || {
                a: tokenSet[i],
                b: tokenSet[i + 1],
                count: 0,
                instances: new Set<number>(),
            };
            entry.count += 1;
            entry.instances.add(j);
            pairs.set(pair, entry);
        }
    }
    return { pairs, tokens };
}

function updatePair(state: InternalTokenState, a: string, b: string, instance: number, v: number): void {
    const pairKey = `${a}${b}`;
    if (state.pairs.has(pairKey)) {
        const pair = state.pairs.get(pairKey)!;
        pair.count += v;
        pair.instances.add(instance);
    } else {
        state.pairs.set(pairKey, { a, b, count: v, instances: new Set([instance]) });
    }
}

function bestPair(state: InternalTokenState): TokenPair | null {
    let bestPair: TokenPair | null = null;
    let maxCount = 0;

    for (const pair of state.pairs.values()) {
        if (pair.count > maxCount) {
            maxCount = pair.count;
            bestPair = pair;
        }
    }

    return bestPair;
}

function mergeTokensAll(tokens: string[][], pair: [string, string]): string[][] {
    return tokens.map((tokenSet) => {
        const newTokenSet: string[] = [];
        for (let i = 0; i < tokenSet.length; i++) {
            if (i < tokenSet.length - 1 && tokenSet[i] === pair[0] && tokenSet[i + 1] === pair[1]) {
                newTokenSet.push(pair[0] + pair[1]);
                i++; // Skip the next token as it has been merged
            } else {
                newTokenSet.push(tokenSet[i]);
            }
        }
        return newTokenSet;
    });
}

function mergeTokens(state: InternalTokenState, pair: TokenPair) {
    pair.instances.forEach((index) => {
        const tokens = state.tokens[index];
        const newTokens: string[] = [];
        for (let i = 0; i < tokens.length; i++) {
            if (i < tokens.length - 1 && tokens[i] === pair.a && tokens[i + 1] === pair.b) {
                const newToken = pair.a + pair.b;
                newTokens.push(newToken);
                if (i > 0) {
                    updatePair(state, tokens[i - 1], pair.a, index, -1);
                    updatePair(state, tokens[i - 1], newToken, index, 1);
                }
                i++; // Skip the next token as it has been merged
                if (i < tokens.length - 1) {
                    updatePair(state, pair.b, tokens[i + 1], index, -1);
                    updatePair(state, newToken, tokens[i + 1], index, 1);
                }
            } else {
                newTokens.push(tokens[i]);
            }
        }
        state.tokens[index] = newTokens;
    });
    // Remove the merged pair from pairs map
    state.pairs.delete(`${pair.a}${pair.b}`);
}

export default class BPE {
    private vocab: Set<string> = new Set();
    private vocabIndex: Map<string, number> = new Map();
    private merges: [string, string][] = [];
    private pretokenMap: Map<string, string[]> = new Map();

    constructor(vocab?: string[], merges?: [string, string][]) {
        if (vocab) {
            // Recreate the merges
            vocab.forEach((v, i) => {
                this.vocab.add(v);
                this.vocabIndex.set(v, i);
            });
        }
        if (merges) {
            this.merges = merges;
        }
    }

    public train(text: string[], vocabSize: number, onUpdate?: (progress: number, vocabSize: number) => void): void {
        const pretokens = text.map((t) => parseTokens(t, true)).flat(1);
        const preTokenSet = new Set<string>(pretokens);

        this.vocab = new Set();
        this.pretokenMap.clear();
        this.merges = [];

        this.vocab.add('<eos>');

        const pretokensArray = Array.from(preTokenSet);
        const tokens = pretokensArray.map((token) => {
            const chars = token.split('');
            return chars.map((c) => {
                this.vocab.add(c);
                return c;
            });
        });

        const state = initPairs(tokens);

        while (this.vocab.size < vocabSize && this.merges.length < vocabSize) {
            //state = initPairs(state.tokens);
            const pair = bestPair(state);
            if (!pair) {
                break; // No more pairs to merge
            }

            this.merges.push([pair.a, pair.b]);
            this.vocab.add(pair.a + pair.b);

            mergeTokens(state, pair);

            if (onUpdate && this.vocab.size % 100 === 0) {
                onUpdate(this.vocab.size / vocabSize, this.vocab.size);
            }
        }

        pretokensArray.forEach((token, i) => {
            const vocabTokens = tokens[i];
            this.pretokenMap.set(token, vocabTokens);
        });

        this.vocabIndex.clear();
        let i = 0;
        for (const v of this.vocab.keys()) {
            this.vocabIndex.set(v, i++);
        }
    }

    public getVocab() {
        return Array.from(this.vocab);
    }

    public getMerges() {
        return this.merges;
    }

    private tokeniseWord(word: string): string[] {
        let tokens = word.split('');
        this.merges.forEach((m) => {
            tokens = mergeTokensAll([tokens], m)[0];
        });
        this.pretokenMap.set(word, tokens);
        return tokens;
    }

    private tokeniseStrings(text: string[]): string[][] {
        return text.map((t) => {
            const pretokens = parseTokens(t, true);
            const tokens = pretokens
                .map((token) => {
                    if (this.pretokenMap.has(token)) {
                        return this.pretokenMap.get(token)!;
                    } else {
                        return this.tokeniseWord(token);
                    }
                })
                .flat(1);
            return tokens;
        });
    }

    public tokenise(text: string[], numeric: true): number[][];
    public tokenise(text: string[]): string[][];
    public tokenise(text: string[], numeric?: boolean): number[][] | string[][] {
        const tokens = this.tokeniseStrings(text);
        if (numeric) {
            return tokens.map((ts) => ts.map((t) => this.vocabIndex.get(t) ?? -1));
        } else {
            return tokens;
        }
    }
}
