import { describe, it } from 'vitest';
import CausalSelfAttention, { KVCache } from './CausalSelfAttention';
import * as tf from '@tensorflow/tfjs';
import RoPECache from './RoPECache';

describe('CausalSelfAttention', () => {
    it('generates a correctly shaped output', ({ expect }) => {
        const layer = new CausalSelfAttention(0, {
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
        layer.dispose();
    });

    it('can generate attention scores', ({ expect }) => {
        const layer = new CausalSelfAttention(0, {
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
        layer.dispose();
    });

    it('can use a KV cache', ({ expect }) => {
        const layer = new CausalSelfAttention(0, {
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

        const input = tf.randomNormal([1, 1, 16]);
        const cache: KVCache = {
            k: tf.randomNormal([1, 2, 2, 8]),
            v: tf.randomNormal([1, 2, 2, 8]),
            length: 2,
            cumulativeLength: 2,
        };
        const { output, presentKV } = layer.call(input, false, false, cache);
        expect(output).toBeInstanceOf(tf.Tensor);
        expect(output.shape).toEqual([1, 1, 16]);
        expect(presentKV).toBeDefined();
        expect(presentKV!.k.shape).toEqual([1, 2, 3, 8]);
        expect(presentKV!.v.shape).toEqual([1, 2, 3, 8]);
        expect(presentKV!.length).toEqual(3);
        expect(presentKV!.cumulativeLength).toEqual(3);
        layer.dispose();
    });

    it('saves and loads weights correctly', ({ expect }) => {
        const input = tf.randomNormal([1, 4, 16]);

        const layer = new CausalSelfAttention(0, {
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

        const newLayer = new CausalSelfAttention(0, {
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

        //layer.dispose();
        newLayer.dispose();
    });

    it('can be trained', async ({ expect }) => {
        const config = {
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
        };
        const ropeCache = new RoPECache(config);
        const layer = new CausalSelfAttention(0, config, ropeCache);

        const input = tf.randomNormal([1, 4, 16]);
        const target = tf.randomNormal([1, 4, 16]);

        layer.call(input, false);

        //const optimizer = tf.train.adam(0.01);

        const f = () => {
            const { output } = layer.call(input, true);
            const loss = tf.losses.meanSquaredError(target, output);
            return loss as tf.Scalar;
        };
        const { value: loss } = tf.variableGrads(f);

        const lossValue = (await loss.data())[0];
        console.log('Final loss:', lossValue);
        expect(lossValue).toBeLessThan(1.5);

        layer.dispose();
    });
});
