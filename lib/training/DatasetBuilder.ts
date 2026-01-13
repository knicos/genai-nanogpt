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

// Training data utilities using TensorFlow.js Dataset API
export class DatasetBuilder {
    public tokenizer: ITokeniser;
    public blockSize: number;
    private pageSize: number;

    constructor(tokenizer: ITokeniser, blockSize: number = 128) {
        this.tokenizer = tokenizer;
        this.blockSize = blockSize;
        this.pageSize = blockSize * PAGE_FACTOR;
    }

    // Create dataset from text files
    public async createTextDataset(
        flatTokens: Uint16Array,
        batchSize: number = 32,
        masked?: Set<number>,
        invertMask?: boolean
    ): Promise<Dataset<{ xs: Tensor; ys: Tensor }>> {
        if (flatTokens.length < this.blockSize + 1) {
            throw new Error(`Not enough tokens (${flatTokens.length}) for block size ${this.blockSize}`);
        }

        if (masked && masked.size > flatTokens.length / this.pageSize / 2) {
            throw new Error('Too many masked pages - would leave insufficient training data');
        }

        // Use generator to avoid storing all sequences in memory
        const gen = function* (this: DatasetBuilder) {
            if (masked && invertMask) {
                const availablePages = Array.from(masked);
                while (true) {
                    const i1 = Math.floor(Math.random() * availablePages.length);
                    const i2 = Math.floor(Math.random() * this.pageSize);
                    const i = availablePages[i1] * this.pageSize + i2;

                    if (i + this.blockSize + 1 > flatTokens.length) {
                        continue; // Skip if out of bounds
                    }

                    const xs = new Int32Array(flatTokens.subarray(i, i + this.blockSize));
                    const ys = new Int32Array(flatTokens.subarray(i + 1, i + this.blockSize + 1));
                    yield { xs, ys };
                }
            } else {
                while (true) {
                    const i = Math.floor(Math.random() * (flatTokens.length - this.blockSize - 1));

                    if (masked) {
                        const maskIndex = Math.floor(i / this.pageSize);
                        const isMasked = masked.has(maskIndex);

                        // TODO: Optimise by calculating valid ranges instead of retrying
                        if ((isMasked && !invertMask) || (!isMasked && invertMask)) {
                            continue; // Skip this sequence
                        }
                    }

                    const xs = new Int32Array(flatTokens.subarray(i, i + this.blockSize));
                    const ys = new Int32Array(flatTokens.subarray(i + 1, i + this.blockSize + 1));
                    yield { xs, ys };
                }
            }
        }.bind(this);

        return generator(gen)
            .batch(batchSize)
            .map((batch) => {
                // Only needed to convert from float32 to int32
                const batchData = batch as { xs: Tensor; ys: Tensor };
                return tidy(() => ({
                    xs: batchData.xs.cast('int32'),
                    ys: batchData.ys.cast('int32'), // this.tf.oneHot(batchData.ys.cast('int32'), this.tokenizer.vocabSize),
                }));
            })
            .prefetch(2); // Smaller prefetch to reduce memory pressure
    }
}
