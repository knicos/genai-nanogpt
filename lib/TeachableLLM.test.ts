import { afterAll, beforeAll, describe, it, vi } from 'vitest';
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import TeachableLLM from './TeachableLLM';
import * as tf from '@tensorflow/tfjs';
import { selectBackend } from './backend';

await tf.setBackend('cpu');

describe('TeachableLLM Tests', () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });
    beforeAll(async () => {
        await selectBackend('webgpu');
    });
    it('can create a model', async ({ expect }) => {
        const model = TeachableLLM.create('char', {
            modelType: 'GenAI_NanoGPT_v2',
            nEmbed: 32,
            nHead: 2,
            nLayer: 1,
            vocabSize: 20,
            blockSize: 6,
            mlpFactor: 4,
        });

        const loadedCB = vi.fn();
        model.on('loaded', loadedCB);

        expect(model).toBeInstanceOf(TeachableLLM);
        expect(model.ready).toBe(false);
        expect(model.loaded).toBe(true);
        await vi.waitFor(() => expect(loadedCB).toHaveBeenCalled());
        model.dispose();
    });

    it('can create multiple if disposed', async ({ expect }) => {
        const model = TeachableLLM.create('char', {
            modelType: 'GenAI_NanoGPT_v2',
            nEmbed: 32,
            nHead: 2,
            nLayer: 1,
            vocabSize: 20,
            blockSize: 6,
            mlpFactor: 4,
        });
        expect(model).toBeInstanceOf(TeachableLLM);
        expect(model.ready).toBe(false);
        model.dispose();

        const newModel = TeachableLLM.create('char', {
            modelType: 'GenAI_NanoGPT_v2',
            nEmbed: 32,
            nHead: 2,
            nLayer: 1,
            vocabSize: 20,
            blockSize: 6,
            mlpFactor: 4,
        });
        expect(newModel).toBeInstanceOf(TeachableLLM);
        expect(newModel.ready).toBe(false);
        newModel.dispose();
    });

    it('has an awaitingTokens status', async ({ expect }) => {
        const model = TeachableLLM.create('char', {
            modelType: 'GenAI_NanoGPT_v2',
            nEmbed: 32,
            nHead: 2,
            nLayer: 1,
            vocabSize: 20,
            blockSize: 6,
            mlpFactor: 4,
        });

        await vi.waitFor(() => expect(model.status).toBe('awaitingTokens'));

        await model.tokeniser.train(['a', 'b', 'c']);

        await vi.waitFor(() => expect(model.status).toBe('ready'));

        model.dispose();
    });
});
