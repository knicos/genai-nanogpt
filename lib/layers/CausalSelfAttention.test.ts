import { describe, it } from 'vitest';
import CausalSelfAttention from './CausalSelfAttention';
import * as tf from '@tensorflow/tfjs';

describe('CausalSelfAttention', () => {
    it('generates a correctly shaped output', ({ expect }) => {
        const layer = new CausalSelfAttention(tf, 0, {
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.0,
            blockSize: 4,
        });

        expect(layer).toBeInstanceOf(CausalSelfAttention);

        const input = tf.randomNormal([1, 4, 16]);
        const output = layer.call(input, false);
        expect(output).toBeInstanceOf(tf.Tensor);
        expect(output.shape).toEqual([1, 4, 16]);
    });

    it('saves and loads weights correctly', ({ expect }) => {
        const input = tf.randomNormal([1, 4, 16]);

        const layer = new CausalSelfAttention(tf, 0, {
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.0,
            blockSize: 4,
        });
        layer.call(input, false); // Initialize the layer

        const weightsMap = new Map<string, tf.Tensor[]>();
        layer.saveWeights(weightsMap);

        const newLayer = new CausalSelfAttention(tf, 0, {
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.0,
            blockSize: 4,
        });
        newLayer.call(input, false); // Initialize the layer
        newLayer.loadWeights(weightsMap);

        const originalOutput = layer.call(input, false);
        const newOutput = newLayer.call(input, false);
        expect(originalOutput.shape).toEqual(newOutput.shape);
        expect(originalOutput.dataSync()).toEqual(newOutput.dataSync());
    });
});
