import '@base/patches/engine';
import { afterEach, describe, it } from 'vitest';
import MLP from './MLP';
import * as tf from '@tensorflow/tfjs';

describe('MLP', () => {
    afterEach(() => {
        tf.disposeVariables();
    });

    it('should call the MLP layer and return a tensor', ({ expect }) => {
        const config = {
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            biasInLayerNorm: false,
            vocabSize: 20,
            dropout: 0.0,
            blockSize: 4,
            mlpFactor: 4,
            useRope: true,
        };
        const mlp = new MLP(0, config);

        const input = tf.randomNormal([1, 4, 16]);
        const output = mlp.call({ training: false }, input) as tf.Tensor;

        expect(output).toBeInstanceOf(tf.Tensor);
        expect(output.shape).toEqual([1, 4, 16]);
    });

    it('should save and load weights correctly', ({ expect }) => {
        const config = {
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            biasInLayerNorm: false,
            vocabSize: 20,
            dropout: 0.0,
            blockSize: 4,
            mlpFactor: 4,
            useRope: true,
        };
        const mlp = new MLP(0, config);
        const input = tf.randomNormal([1, 4, 16]);
        mlp.call({ training: false }, input); // Initialize the layer

        const weightsMap = new Map<string, tf.Tensor[]>();
        mlp.saveWeights(weightsMap);
        const originalOutput = mlp.call({ training: false }, input) as tf.Tensor;

        mlp.dispose();

        const newMlp = new MLP(0, config);
        newMlp.call({ training: false }, input); // Initialize the layer
        newMlp.loadWeights(weightsMap);

        const newOutput = newMlp.call({ training: false }, input) as tf.Tensor;
        expect(originalOutput.shape).toEqual(newOutput.shape);
        expect(originalOutput.dataSync()).toEqual(newOutput.dataSync());

        newMlp.dispose();
        input.dispose();
        originalOutput.dispose();
        newOutput.dispose();
    });
});
