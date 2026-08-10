import { describe, it, vi } from 'vitest';
import { DatasetBuilder, DatasetState, flattenTokens, flattenTokensWithMask, moveToNext } from './DatasetBuilder';
import * as tf from '@tensorflow/tfjs';
import type { Conversation, ITokeniser } from '../tokeniser/type';
import { TokenStore } from './tasks/TokenStore';

await tf.setBackend('cpu');

describe('DatasetBuilder', () => {
    it('should create a dataset with conversation data', async ({ expect }) => {
        const mockTokenizer = {
            vocabSize: 256,
            encodeConversation: vi.fn((conversation: Conversation[]) =>
                conversation.map((msg) => msg.content.split('').map((c: string) => c.charCodeAt(0))).flat()
            ),
        } as unknown as ITokeniser;
        const blockSize = 10;

        // Create instance of DatasetBuilder
        const datasetBuilder = new DatasetBuilder(mockTokenizer, blockSize);

        // Test createTextDataset method
        const textData: Conversation[][] = [
            [
                { role: 'user', content: 'hello' },
                { role: 'assistant', content: 'hi there' },
            ],
            [
                { role: 'user', content: 'how are you?' },
                { role: 'assistant', content: 'I am fine' },
            ],
        ];
        const allTokens = flattenTokens(textData, mockTokenizer);
        const store = new TokenStore('mockTokenizer', 'mockDataset');
        await store.appendShard(allTokens);
        const { dataset } = await datasetBuilder.createTextDataset(store, {
            batchSize: 2,
        });

        // Assertions
        expect(dataset).toBeDefined();

        // Check if dataset has the expected structure
        const iterator = await dataset.iterator();
        const firstBatch = await iterator.next();
        const value: { xs: tf.Tensor; ys: tf.Tensor } = firstBatch.value;
        expect(value).toBeDefined();
        expect(value.xs.shape).toEqual([2, blockSize]);
        expect(value.ys.shape).toEqual([2, blockSize]); // , mockTokenizer.vocabSize]);

        for (let i = 0; i < 10; i++) {
            const nextBatch = await iterator.next();
            if (nextBatch.done) break;
        }

        expect(mockTokenizer.encodeConversation).toHaveBeenCalledTimes(2);
        expect(mockTokenizer.encodeConversation).toHaveBeenCalledWith(textData[0]);
        expect(mockTokenizer.encodeConversation).toHaveBeenCalledWith(textData[1]);
    });

    it('should support masking', async ({ expect }) => {
        const mockTokenizer = {
            vocabSize: 256,
            encodeConversation: vi.fn((conversation: Conversation[], _?: boolean, masking?: boolean) => {
                if (masking) {
                    return {
                        tokens: conversation
                            .map((msg) => msg.content.split('').map((c: string) => c.charCodeAt(0)))
                            .flat(),
                        mask: conversation
                            .map((msg) => msg.content.split('').map(() => (msg.role === 'user' ? false : true)))
                            .flat(),
                    };
                } else {
                    return conversation.map((msg) => msg.content.split('').map((c: string) => c.charCodeAt(0))).flat();
                }
            }),
        } as unknown as ITokeniser;
        const blockSize = 10;

        // Create instance of DatasetBuilder
        const datasetBuilder = new DatasetBuilder(mockTokenizer, blockSize);

        // Test createTextDataset method
        const textData: Conversation[][] = [
            [
                { role: 'user', content: 'hello' },
                { role: 'assistant', content: 'hi there' },
            ],
            [
                { role: 'user', content: 'how are you?' },
                { role: 'assistant', content: 'I am fine' },
            ],
        ];
        const allTokens = flattenTokensWithMask(textData, mockTokenizer);
        console.log('All Tokens:', allTokens.tokens);
        console.log('All Masks:', allTokens.mask);
        const store = new TokenStore('mockTokenizer', 'mockDataset');
        await store.appendShard(allTokens.tokens, allTokens.mask);
        const { dataset } = await datasetBuilder.createTextDataset(store, {
            batchSize: 2,
        });

        // Assertions
        expect(dataset).toBeDefined();

        // Check if dataset has the expected structure
        const iterator = await dataset.iterator();
        const firstBatch = await iterator.next();
        const value: { xs: tf.Tensor; ys: tf.Tensor } = firstBatch.value;
        expect(value).toBeDefined();
        expect(value.xs.shape).toEqual([2, blockSize]);
        expect(value.ys.shape).toEqual([2, blockSize]); // , mockTokenizer.vocabSize]);

        // Check that some tokens are indeed masked in ys
        const ysData = (await value.ys.array()) as number[][];
        console.log('YS Data:', ysData);
        const hasMaskedToken = ysData.some((row) => row.some((token) => token === 0xffff));
        expect(hasMaskedToken).toBe(true);

        for (let i = 0; i < 10; i++) {
            const nextBatch = await iterator.next();
            if (nextBatch.done) break;
        }

        expect(mockTokenizer.encodeConversation).toHaveBeenCalledTimes(2);
        expect(mockTokenizer.encodeConversation).toHaveBeenCalledWith(textData[0], false, true);
        expect(mockTokenizer.encodeConversation).toHaveBeenCalledWith(textData[1], false, true);
    });
});

describe('moveToNext', () => {
    it('should move to the next shard and reset step', async ({ expect }) => {
        const state: DatasetState = {
            shuffledShards: new Uint32Array([0, 1]),
            shuffledIndexes: new Uint32Array([0, 1, 2]),
            lastShardIndexes: new Uint32Array([0, 1, 2]),
            currentShard: new Uint16Array([1, 2, 3]),
            nextShard: new Uint16Array([4, 5, 6]),
            currentMask: null,
            nextMask: null,
            shardIndex: 0,
            step: 2,
        };

        const mockStore = {
            hasMask: vi.fn(() => false),
            getShard: vi.fn(async (index: number) => {
                if (index === 0) return new Uint16Array([1, 2, 3]);
                if (index === 1) return new Uint16Array([4, 5, 6]);
                throw new Error('Invalid shard index');
            }),
        } as unknown as TokenStore;

        await moveToNext(state, mockStore);

        expect(state.step).toBe(0);
        expect(state.shardIndex).toBe(1);
        expect(state.currentShard).toEqual(new Uint16Array([4, 5, 6]));
    });

    it('start again after last shard', async ({ expect }) => {
        const state: DatasetState = {
            shuffledShards: new Uint32Array([0, 1]),
            shuffledIndexes: new Uint32Array([0, 1, 2]),
            lastShardIndexes: new Uint32Array([0, 1, 2]),
            currentShard: new Uint16Array([4, 5, 6]),
            nextShard: null,
            currentMask: null,
            nextMask: null,
            shardIndex: 1,
            step: 2,
        };

        const mockStore = {
            hasMask: vi.fn(() => false),
            getShard: vi.fn(async (index: number) => {
                if (index === 0) return new Uint16Array([1, 2, 3]);
                if (index === 1) return new Uint16Array([4, 5, 6]);
                throw new Error('Invalid shard index');
            }),
        } as unknown as TokenStore;

        await moveToNext(state, mockStore, true);

        expect(state.step).toBe(0);
        expect(state.shardIndex).toBe(0);
        expect(state.currentShard).toEqual(new Uint16Array([1, 2, 3]));
    });
});
