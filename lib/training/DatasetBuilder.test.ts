import { describe, it, vi } from 'vitest';
import { DatasetBuilder, flattenTokens } from './DatasetBuilder';
import * as tf from '@tensorflow/tfjs';
import type { Conversation, ITokeniser } from '../tokeniser/type';

await tf.setBackend('cpu');

describe('DatasetBuilder', () => {
    it('should create a dataset with conversation data', async ({ expect }) => {
        const mockTokenizer = {
            vocabSize: 256,
            encodeConversation: vi.fn(async (conversation: Conversation[]) =>
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
        const allTokens = new Uint16Array(await flattenTokens(textData, mockTokenizer));
        const { dataset } = await datasetBuilder.createTextDataset(allTokens, 2);

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

    it('work with provided indexes', async ({ expect }) => {
        const mockTokenizer = {
            vocabSize: 256,
            encodeConversation: vi.fn(async (conversation: Conversation[]) =>
                conversation.map((msg) => msg.content.split('').map((c: string) => c.charCodeAt(0))).flat()
            ),
        } as unknown as ITokeniser;
        const blockSize = 1;

        // Create instance of DatasetBuilder
        const datasetBuilder = new DatasetBuilder(mockTokenizer, blockSize);

        // Test createTextDataset method with a single text input
        const textData: Conversation[] = [{ role: 'user', content: 'hello world hello world hello world hello world' }];
        const allTokens = new Uint16Array(await flattenTokens([textData], mockTokenizer));
        const indexes = [0, 6, 12, 18]; // Only take the first token of each "hello"
        const { dataset, state } = await datasetBuilder.createTextDataset(allTokens, 2, indexes);

        // Assertions
        expect(dataset).toBeDefined();

        const iterator = await dataset.iterator();
        const firstBatch = await iterator.next();
        const value: { xs: tf.Tensor; ys: tf.Tensor } = firstBatch.value;
        expect(value).toBeDefined();
        expect(value.xs.shape).toEqual([2, blockSize]);
        expect(value.ys.shape).toEqual([2, blockSize]); //, mockTokenizer.vocabSize]);

        // Check that the tokens correspond to the provided indexes
        const xsData = (await value.xs.array()) as number[][];
        const ysData = (await value.ys.array()) as number[][];
        expect(xsData[0][0]).toBe('h'.charCodeAt(0)); // 'hello' first token
        expect(xsData[1][0]).toBe('w'.charCodeAt(0)); // 'world' first token
        expect(ysData[0][0]).toBe('e'.charCodeAt(0)); // 'hello' second token
        expect(ysData[1][0]).toBe('o'.charCodeAt(0)); // 'world' second token

        for (let i = 0; i < 4; i++) {
            const nextBatch = await iterator.next();
            if (nextBatch.done) break;
        }

        expect(state.step).toBe(0); // Should reset after going through all indexes
    });
});
