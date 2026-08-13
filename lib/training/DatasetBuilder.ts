import { Tensor, tidy } from '@tensorflow/tfjs-core';
import type { Conversation, ITokeniser } from '../tokeniser/type';
import { Dataset, generator } from '@tensorflow/tfjs-data';
import { TokenStore } from './tasks/TokenStore';

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

export function shuffle(array: Uint32Array | Uint16Array): Uint32Array | Uint16Array {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

export interface DatasetState {
    shuffledShards: Uint16Array;
    shuffledIndexes: Uint16Array;
    lastShardIndexes: Uint16Array;
    currentShard: Uint16Array | null;
    nextShard: Uint16Array | null;
    currentMask: Uint8Array | null;
    nextMask: Uint8Array | null;
    shardIndex: number;
    step: number;
}

export async function moveToNext(state: DatasetState, store: TokenStore, noShuffle?: boolean) {
    state.step += 1;

    const indexes =
        state.shardIndex === state.shuffledShards.length - 1 ? state.lastShardIndexes : state.shuffledIndexes;

    // End of shard
    if (state.step >= indexes.length) {
        state.step = 0;
        state.shardIndex += 1;
        // End of all shards
        if (state.shardIndex >= state.shuffledShards.length) {
            state.shardIndex = 0;
            if (!noShuffle) {
                shuffle(state.shuffledShards);
                shuffle(state.shuffledIndexes);
                shuffle(state.lastShardIndexes);
            }
        }

        if (state.nextShard) {
            state.currentShard = state.nextShard;
            state.nextShard = null;
        } else {
            state.currentShard = await store.getShard(state.shuffledShards[state.shardIndex]);
        }
        // Preload next shard
        const nextShardIndex = (state.shardIndex + 1) % state.shuffledShards.length;
        store.getShard(state.shuffledShards[nextShardIndex]).then((shard) => {
            state.nextShard = shard;
        });

        if (store.hasMask()) {
            if (state.nextMask) {
                state.currentMask = state.nextMask;
                state.nextMask = null;
            } else {
                state.currentMask = (await store.getMask(state.shuffledShards[state.shardIndex])) ?? null;
            }
            // Preload next mask
            const nextMaskIndex = (state.shardIndex + 1) % state.shuffledShards.length;
            store.getMask(state.shuffledShards[nextMaskIndex]).then((mask) => {
                state.nextMask = mask ?? null;
            });
        }
    }
}

interface DatasetOptions {
    batchSize: number;
    noShuffle?: boolean;
    ignoreIndex?: number;
    shuffleFirst?: boolean;
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
        store: TokenStore,
        options?: DatasetOptions
    ): Promise<{ dataset: Dataset<{ xs: Tensor; ys: Tensor }>; state: DatasetState }> {
        const { batchSize = 32, noShuffle = false, ignoreIndex = 0xffff } = options || {};
        const totalTokens = store.getTokenCount();

        if (totalTokens < this.blockSize + 1) {
            throw new Error(`Not enough tokens (${totalTokens}) for block size ${this.blockSize}`);
        }

        const blocksPerShard = Math.ceil(store.shardSize / this.blockSize);

        const state: DatasetState = {
            shuffledShards: new Uint16Array(store.getShardCount()),
            shuffledIndexes: new Uint16Array(blocksPerShard),
            lastShardIndexes: new Uint16Array(
                Math.ceil(store.getShardLength(store.getShardCount() - 1) / this.blockSize)
            ),
            currentMask: null,
            nextMask: null,
            currentShard: null,
            nextShard: null,
            shardIndex: 0,
            step: 0,
        };

        for (let i = 0; i < state.shuffledShards.length; i++) {
            state.shuffledShards[i] = i;
        }
        for (let i = 0; i < state.shuffledIndexes.length; i++) {
            state.shuffledIndexes[i] = i;
        }
        for (let i = 0; i < state.lastShardIndexes.length; i++) {
            state.lastShardIndexes[i] = i;
        }

        if (options?.shuffleFirst) {
            shuffle(state.shuffledShards);
            shuffle(state.shuffledIndexes);
            shuffle(state.lastShardIndexes);
        }

        // Await current shard
        state.currentShard = await store.getShard(state.shuffledShards[state.shardIndex]);
        // Preload next shard
        if (state.shardIndex + 1 < state.shuffledShards.length) {
            store.getShard(state.shuffledShards[state.shardIndex + 1]).then((shard) => {
                state.nextShard = shard;
            });
        }

        if (store.hasMask()) {
            state.currentMask = (await store.getMask(state.shuffledShards[state.shardIndex])) ?? null;
            // Preload next mask
            if (state.shardIndex + 1 < state.shuffledShards.length) {
                store.getMask(state.shuffledShards[state.shardIndex + 1]).then((mask) => {
                    state.nextMask = mask ?? null;
                });
            }
        }

        // Use generator to avoid storing all sequences in memory
        const gen = async function* (this: DatasetBuilder) {
            while (true) {
                const indexes =
                    state.shardIndex === state.shuffledShards.length - 1
                        ? state.lastShardIndexes
                        : state.shuffledIndexes;
                const step = indexes[state.step];
                let i = step * this.blockSize;
                const flatTokens = state.currentShard;
                const mask = state.currentMask;

                const move = moveToNext(state, store, noShuffle);

                if (!flatTokens) {
                    break;
                }
                if (i + this.blockSize + 1 > flatTokens.length) {
                    i = flatTokens.length - this.blockSize - 1;
                }

                const xs = new Int32Array(flatTokens.slice(i, i + this.blockSize));
                const ys = new Int32Array(flatTokens.slice(i + 1, i + this.blockSize + 1));

                if (mask) {
                    let count = 0;
                    const flatMask = mask.slice(i + 1, i + this.blockSize + 1);
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

                await move;
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
                        ys: batchData.ys.cast('int32'),
                    }));
                })
                .prefetch(2), // Smaller prefetch to reduce memory pressure
            state,
        };
    }
}
