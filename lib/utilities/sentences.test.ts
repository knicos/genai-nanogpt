import { afterAll, describe, it } from 'vitest';
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';
import TeachableLLM from '@base/TeachableLLM';
import { sentenceEmbeddings } from './sentences';

describe('Sentence embeddings', { timeout: 60000 }, () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('creates an embedding from one short sentence', async ({ expect }) => {
        await selectBackend('webgpu');

        const sentences = ['Hello world!'];

        const model = TeachableLLM.create('char', {
            blockSize: 128,
            nEmbed: 128,
            nLayer: 2,
            nHead: 4,
        });

        await model.trainTokeniser(sentences);

        const embeddings = await sentenceEmbeddings(model, sentences);
        expect(embeddings.length).toBe(1);
        expect(embeddings[0].length).toBe(128);

        model.dispose();
    });

    it('creates an embedding from one long sentence', async ({ expect }) => {
        await selectBackend('webgpu');

        const sentences = [
            'Hello world, this is a much longer sentence designed to test the sentence embedding functionality of the TeachableLLM model. It should be truncated or handled appropriately to fit within the context length of the model.',
        ];

        const model = TeachableLLM.create('char', {
            blockSize: 128,
            nEmbed: 128,
            nLayer: 2,
            nHead: 4,
        });

        await model.trainTokeniser(sentences);

        const embeddings = await sentenceEmbeddings(model, sentences);
        expect(embeddings.length).toBe(1);
        expect(embeddings[0].length).toBe(128);

        model.dispose();
    });

    it('creates embeddings from many sentences', async ({ expect }) => {
        await selectBackend('webgpu');

        const sentences = [
            'Hello world, this is a sentence',
            'The quick brown fox jumps over the lazy dog',
            'Artificial intelligence is transforming the world',
            'TeachableLLM makes it easy to create custom language models',
            'Unit testing is essential for reliable software development',
            'WebGPU provides powerful capabilities for machine learning in the browser',
        ];

        const model = TeachableLLM.create('char', {
            blockSize: 128,
            nEmbed: 128,
            nLayer: 2,
            nHead: 4,
        });

        await model.trainTokeniser(sentences);

        const embeddings = await sentenceEmbeddings(model, sentences);
        expect(embeddings.length).toBe(sentences.length);
        expect(embeddings[0].length).toBe(128);

        model.dispose();
    });
});
