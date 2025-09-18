import { afterEach, describe, it } from 'vitest';
import TiedEmbedding from './TiedEmbedding';
import * as tf from '@tensorflow/tfjs';

describe('TiedEmbedding', () => {
    afterEach(() => {
        tf.disposeVariables();
    });

    it('should embed inputs correctly', ({ expect }) => {
        const config = { vocabSize: 100, embedDim: 64 };
        const tiedEmbedding = new TiedEmbedding(config);
        const input = tf.tensor1d([1, 2, 3], 'int32');
        const output = tiedEmbedding.embed(input);
        expect(output.shape).toEqual([3, 64]); // Shape should match [batch_size, embed_dim]
    });

    it('should project inputs correctly', ({ expect }) => {
        const config = { vocabSize: 100, embedDim: 64 };
        const tiedEmbedding = new TiedEmbedding(config);
        const input = tf.randomNormal([1, 4, 64]);
        const output = tiedEmbedding.project(input);
        expect(output.shape).toEqual([1, 4, 100]); // Shape should match [batch_size, sequence_length, vocab_size]
    });

    it('should embed and project with the same weights', ({ expect }) => {
        const config = { vocabSize: 20, embedDim: 8 };
        const tiedEmbedding = new TiedEmbedding(config);
        const input = tf.tensor1d([1, 2, 3], 'int32');
        const embedded = tiedEmbedding.embed(input);
        const projected = tiedEmbedding.project(embedded);

        // This is extremely unrealiable
        const { indices } = tf.topk(projected, 4);
        const indicesData = indices.arraySync() as number[][];
        const inputData = input.arraySync();
        expect(indicesData[0]).toContain(inputData[0]);
        expect(indicesData[1]).toContain(inputData[1]);
        expect(indicesData[2]).toContain(inputData[2]);
    });

    it('restores from saved weights', ({ expect }) => {
        const config = { vocabSize: 10, embedDim: 5 };
        const tiedEmbedding = new TiedEmbedding(config);
        const input = tf.tensor1d([1, 2, 3], 'int32');
        tiedEmbedding.embed(input); // Initialize the layer

        const weights = tiedEmbedding.getWeights();

        const newTiedEmbedding = new TiedEmbedding(config, 'new_tied_embedding');
        newTiedEmbedding.setWeights(weights);

        const originalOutput = tiedEmbedding.embed(input);
        const newOutput = newTiedEmbedding.embed(input);

        expect(originalOutput.shape).toEqual(newOutput.shape);
        expect(originalOutput.dataSync()).toEqual(newOutput.dataSync());
    });
});
