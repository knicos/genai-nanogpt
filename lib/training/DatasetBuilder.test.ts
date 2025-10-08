import { describe, it, vi } from 'vitest';
import { DatasetBuilder, flattenTokens } from './DatasetBuilder';
import * as tf from '@tensorflow/tfjs';
import type { ITokeniser } from '../tokeniser/type';

await tf.setBackend('cpu');

describe('DatasetBuilder', () => {
    it('should create a dataset from text data', async ({ expect }) => {
        const mockTokenizer = {
            vocabSize: 256,
            encode: vi.fn(async (text: string) => text.split('').map((c) => c.charCodeAt(0))),
        } as unknown as ITokeniser;
        const blockSize = 10;

        // Create instance of DatasetBuilder
        const datasetBuilder = new DatasetBuilder(mockTokenizer, blockSize);

        // Test createTextDataset method
        const textData = ['hello world', 'this is a test'];
        const allTokens = await flattenTokens(textData, mockTokenizer);
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

        expect(mockTokenizer.encode).toHaveBeenCalledTimes(2);
        expect(mockTokenizer.encode).toHaveBeenCalledWith('hello world');
        expect(mockTokenizer.encode).toHaveBeenCalledWith('this is a test');

        console.log('Dataset created successfully:', value.xs.arraySync());
    });

    it('supports a single text input', async ({ expect }) => {
        const mockTokenizer = {
            vocabSize: 256,
            encode: vi.fn(async (text: string) => text.split('').map((c) => c.charCodeAt(0))),
        } as unknown as ITokeniser;
        const blockSize = 10;

        // Create instance of DatasetBuilder
        const datasetBuilder = new DatasetBuilder(mockTokenizer, blockSize);

        // Test createTextDataset method with a single text input
        const textData = ['hello world'];
        const allTokens = await flattenTokens(textData, mockTokenizer);
        const dataset = await datasetBuilder.createTextDataset(allTokens, 2);

        // Assertions
        expect(dataset).toBeDefined();

        const iterator = await dataset.iterator();
        const firstBatch = await iterator.next();
        const value: { xs: tf.Tensor; ys: tf.Tensor } = firstBatch.value;
        expect(value).toBeDefined();
        expect(value.xs.shape).toEqual([2, blockSize]);
        expect(value.ys.shape).toEqual([2, blockSize]); //, mockTokenizer.vocabSize]);
    });

    it('masks validation pages', async ({ expect }) => {
        const mockTokenizer = {
            vocabSize: 256,
            encode: vi.fn(async (text: string) => text.split('').map((c) => c.charCodeAt(0))),
        } as unknown as ITokeniser;
        const blockSize = 1;

        // Create instance of DatasetBuilder
        const datasetBuilder = new DatasetBuilder(mockTokenizer, blockSize);

        // Test createTextDataset method with a single text input
        const textData = ['hello world hello world hello world hello world'];
        const allTokens = await flattenTokens(textData, mockTokenizer);
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
