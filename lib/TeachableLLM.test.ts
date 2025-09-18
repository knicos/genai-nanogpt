import { describe, it, vi } from 'vitest';
import TeachableLLM from './TeachableLLM';
import * as tf from '@tensorflow/tfjs';

await tf.setBackend('cpu');

describe('TeachableLLM Tests', () => {
    it('can create a model', async ({ expect }) => {
        const model = TeachableLLM.create('char', {
            nEmbed: 32,
            nHead: 2,
            nLayer: 1,
            vocabSize: 20,
            blockSize: 6,
            dropout: 0.1,
            biasInLinear: false,
            biasInLayerNorm: false,
            mlpFactor: 4,
        });
        expect(model).toBeInstanceOf(TeachableLLM);
        expect(model.ready).toBe(false);
        model.dispose();
    });

    it('can create multiple if disposed', async ({ expect }) => {
        const model = TeachableLLM.create('char', {
            nEmbed: 32,
            nHead: 2,
            nLayer: 1,
            vocabSize: 20,
            blockSize: 6,
            dropout: 0.1,
            biasInLinear: false,
            biasInLayerNorm: false,
            mlpFactor: 4,
        });
        expect(model).toBeInstanceOf(TeachableLLM);
        expect(model.ready).toBe(false);
        model.dispose();

        const newModel = TeachableLLM.create('char', {
            nEmbed: 32,
            nHead: 2,
            nLayer: 1,
            vocabSize: 20,
            blockSize: 6,
            dropout: 0.1,
            biasInLinear: false,
            biasInLayerNorm: false,
            mlpFactor: 4,
        });
        expect(newModel).toBeInstanceOf(TeachableLLM);
        expect(newModel.ready).toBe(false);
        newModel.dispose();
    });

    it('has an awaitingTokens status', async ({ expect }) => {
        const model = TeachableLLM.create('char', {
            nEmbed: 32,
            nHead: 2,
            nLayer: 1,
            vocabSize: 20,
            blockSize: 6,
            dropout: 0.1,
            biasInLinear: false,
            biasInLayerNorm: false,
            mlpFactor: 4,
        });

        await vi.waitFor(() => expect(model.status).toBe('awaitingTokens'));

        await model.tokeniser.train(['a', 'b', 'c']);

        await vi.waitFor(() => expect(model.status).toBe('ready'));

        model.dispose();
    });
});
