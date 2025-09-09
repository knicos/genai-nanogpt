import { describe, it } from 'vitest';
import CausalSelfAttention from './CausalSelfAttention';
import * as tf from '@tensorflow/tfjs';

describe('CausalSelfAttention', () => {
    it('generates a correctly shaped output', ({ expect }) => {
        const layer = new CausalSelfAttention(tf, 0, {
            biasInLayerNorm: false,
            vocabSize: 20,
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.0,
            blockSize: 4,
            mlpFactor: 4,
            useRope: true,
        });

        expect(layer).toBeInstanceOf(CausalSelfAttention);

        const input = tf.randomNormal([1, 4, 16]);
        const { output } = layer.call(input, false);
        expect(output).toBeInstanceOf(tf.Tensor);
        expect(output.shape).toEqual([1, 4, 16]);
    });

    it('can generate attention scores', ({ expect }) => {
        const layer = new CausalSelfAttention(tf, 0, {
            biasInLayerNorm: false,
            vocabSize: 20,
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.0,
            blockSize: 4,
            mlpFactor: 4,
            useRope: true,
        });

        const input = tf.randomNormal([1, 4, 16]);
        const { attention } = layer.call(input, false, true);
        expect(attention).toBeInstanceOf(tf.Tensor);
        expect(attention!.shape).toEqual([1, 4, 4]);

        console.log('Attention', attention!.toString());
    });

    it('saves and loads weights correctly', ({ expect }) => {
        const input = tf.randomNormal([1, 4, 16]);

        const layer = new CausalSelfAttention(tf, 0, {
            biasInLayerNorm: false,
            vocabSize: 20,
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.0,
            blockSize: 4,
            mlpFactor: 4,
            useRope: true,
        });
        layer.call(input, false); // Initialize the layer

        const weightsMap = new Map<string, tf.Tensor[]>();
        layer.saveWeights(weightsMap);

        const newLayer = new CausalSelfAttention(tf, 0, {
            biasInLayerNorm: false,
            vocabSize: 20,
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.0,
            blockSize: 4,
            mlpFactor: 4,
            useRope: true,
        });
        newLayer.call(input, false); // Initialize the layer
        newLayer.loadWeights(weightsMap);

        const { output: originalOutput } = layer.call(input, false);
        const { output: newOutput } = newLayer.call(input, false);
        expect(originalOutput.shape).toEqual(newOutput.shape);
        expect(originalOutput.dataSync()).toEqual(newOutput.dataSync());
    });
});
