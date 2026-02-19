import { Tensor, tidy } from '@tensorflow/tfjs-core';
import type { Conversation, ITokeniser } from '../tokeniser/type';
import { Dataset, generator } from '@tensorflow/tfjs-data';

export const PAGE_FACTOR = 8;

export async function flattenTokens(textData: Conversation[][], tokenizer: ITokeniser): Promise<number[]> {
    // Process ALL text into one token array first
    const tokenisedTexts = await Promise.all(textData.map((text) => tokenizer.encodeConversation(text)));

    const flatTokens = tokenisedTexts.flat();
    return flatTokens;
}

export function shuffle(array: number[]): number[] {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

export interface DatasetState {
    shuffledIndexes: number[];
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
        flatTokens: Uint16Array,
        batchSize = 32,
        indexes?: number[]
    ): Promise<{ dataset: Dataset<{ xs: Tensor; ys: Tensor }>; state: DatasetState }> {
        if (flatTokens.length < this.blockSize + 1) {
            throw new Error(`Not enough tokens (${flatTokens.length}) for block size ${this.blockSize}`);
        }

        const state: DatasetState = {
            shuffledIndexes: [],
            step: 0,
        };

        if (indexes) {
            state.shuffledIndexes = indexes;
            // shuffle(state.shuffledIndexes);
        } else {
            state.shuffledIndexes = Array.from({ length: flatTokens.length }, (_, i) => i);
            shuffle(state.shuffledIndexes);
        }

        // Use generator to avoid storing all sequences in memory
        const gen = function* (this: DatasetBuilder) {
            while (true) {
                const i = state.shuffledIndexes[state.step++];

                if (state.step >= state.shuffledIndexes.length) {
                    state.step = 0;
                    shuffle(state.shuffledIndexes);
                }

                if (i + this.blockSize + 1 > flatTokens.length) {
                    continue; // Skip if out of bounds
                }

                const xs = new Int32Array(flatTokens.subarray(i, i + this.blockSize));
                const ys = new Int32Array(flatTokens.subarray(i + 1, i + this.blockSize + 1));
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
