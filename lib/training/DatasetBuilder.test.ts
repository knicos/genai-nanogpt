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
        const dataset = await datasetBuilder.createTextDataset(allTokens, 2);

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

    it('masks validation pages', async ({ expect }) => {
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
        const maskSet = new Set<number>();
        maskSet.add(0); // Mask the first page
        const dataset = await datasetBuilder.createTextDataset(allTokens, 2, maskSet, false);

        // Assertions
        expect(dataset).toBeDefined();

        const iterator = await dataset.iterator();
        const firstBatch = await iterator.next();
        const value: { xs: tf.Tensor; ys: tf.Tensor } = firstBatch.value;
        expect(value).toBeDefined();
        expect(value.xs.shape).toEqual([2, blockSize]);
        expect(value.ys.shape).toEqual([2, blockSize]); //, mockTokenizer.vocabSize]);
    });
});
