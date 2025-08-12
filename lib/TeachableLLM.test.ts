import { describe, it } from 'vitest';
import TeachableLLM from './TeachableLLM';
import * as tf from '@tensorflow/tfjs';

describe('TeachableLLM Tests', () => {
    it('can create a model', async ({ expect }) => {
        const model = TeachableLLM.create(tf, {
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
        const model = TeachableLLM.create(tf, {
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

        const newModel = TeachableLLM.create(tf, {
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
});
