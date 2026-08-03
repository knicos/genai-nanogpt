import { Tensor, tidy } from '@tensorflow/tfjs-core';
import type { Conversation, ITokeniser } from '../tokeniser/type';
import { Dataset, generator } from '@tensorflow/tfjs-data';
import { sliceUint16Shards, sliceUint8Shards } from '@base/utilities/tokens';

export function flattenTokens(textData: Conversation[][], tokenizer: ITokeniser): Uint16Array {
    // Process ALL text into one token array first
    const tokenisedTexts = textData.map((text) => tokenizer.encodeConversation(text));

    const flatTokens = tokenisedTexts.flat();
    return new Uint16Array(flatTokens);
}

export function flattenTokensWithMask(
    textData: Conversation[][],
    tokenizer: ITokeniser
): { tokens: Uint16Array; mask: Uint8Array } {
    // Process ALL text into one token array first
    const tokenisedTexts = textData.map((text) => tokenizer.encodeConversation(text, false, true));

    const flatTokens = tokenisedTexts.map((t) => t.tokens).flat();
    const mask = tokenisedTexts.map((t) => t.mask).flat();
    return { tokens: new Uint16Array(flatTokens), mask: new Uint8Array(mask.map((m) => (m ? 1 : 0))) };
}

export function shuffle(array: Uint32Array): Uint32Array {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

export interface DatasetState {
    shuffledIndexes: Uint32Array;
    step: number;
}

// Training data utilities using TensorFlow.js Dataset API
export class DatasetBuilder {
    public tokenizer: ITokeniser;
    public blockSize: number;

    constructor(tokenizer: ITokeniser, blockSize = 128) {
        this.tokenizer = tokenizer;
        this.blockSize = blockSize;
    }

    // Create dataset from text files
    public async createTextDataset(
        flatTokens: Uint16Array[],
        batchSize = 32,
        indexes?: Uint32Array,
        mask?: Uint8Array[],
        ignoreIndex = 0xffff
    ): Promise<{ dataset: Dataset<{ xs: Tensor; ys: Tensor }>; state: DatasetState }> {
        const totalTokens = flatTokens.reduce((sum, tokens) => sum + tokens.length, 0);

        if (totalTokens < this.blockSize + 1) {
            throw new Error(`Not enough tokens (${totalTokens}) for block size ${this.blockSize}`);
        }

        const totalBlocks = Math.ceil(totalTokens / this.blockSize);

        const state: DatasetState = {
            shuffledIndexes: new Uint32Array(totalBlocks),
            step: 0,
        };

        // Note: Don't actually shuffle on the first epoch to allow curriculum learning. We'll shuffle after the first epoch.
        if (indexes) {
            state.shuffledIndexes = indexes;
        } else {
            state.shuffledIndexes = new Uint32Array(totalBlocks);
            for (let i = 0; i < totalBlocks; i++) {
                state.shuffledIndexes[i] = i;
            }
        }

        // Use generator to avoid storing all sequences in memory
        const gen = function* (this: DatasetBuilder) {
            while (true) {
                const i = state.shuffledIndexes[state.step++] * this.blockSize;

                if (state.step >= state.shuffledIndexes.length) {
                    state.step = 0;
                    shuffle(state.shuffledIndexes);
                }

                if (i + this.blockSize + 1 > totalTokens) {
                    continue; // Skip if out of bounds
                }

                const xs = new Int32Array(sliceUint16Shards(flatTokens, i, i + this.blockSize));
                const ys = new Int32Array(sliceUint16Shards(flatTokens, i + 1, i + this.blockSize + 1));

                if (mask) {
                    let count = 0;
                    const flatMask = sliceUint8Shards(mask, i + 1, i + this.blockSize + 1);
                    for (let j = 0; j < ys.length; j++) {
                        if (flatMask[j] === 0) {
                            ys[j] = ignoreIndex;
                            count++;
                        }
                    }
                    if (count === ys.length) {
                        continue; // Skip if all tokens are masked
                    }
                }

                yield { xs, ys };
            }
        }.bind(this);

        return {
            dataset: generator(gen)
                .batch(batchSize)
                .map((batch) => {
                    // Only needed to convert from float32 to int32
                    const batchData = batch as { xs: Tensor; ys: Tensor };
                    return tidy(() => ({
                        xs: batchData.xs.cast('int32'),
                        ys: batchData.ys.cast('int32'), // this.tf.oneHot(batchData.ys.cast('int32'), this.tokenizer.vocabSize),
                    }));
                })
                .prefetch(2), // Smaller prefetch to reduce memory pressure
            state,
        };
    }
}
