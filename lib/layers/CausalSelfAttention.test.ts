import { describe, it } from 'vitest';
import CausalSelfAttention, { KVCache } from './CausalSelfAttention';
import * as tf from '@tensorflow/tfjs';
import RoPECache from './RoPECache';
import { afterEach } from 'node:test';

describe('CausalSelfAttention', () => {
    afterEach(() => {
        tf.disposeVariables();
    });

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
        const output = layer.call({ training: false }, input) as tf.Tensor;
        expect(output).toBeInstanceOf(tf.Tensor);
        expect(output.shape).toEqual([1, 4, 16]);
        layer.dispose();
    });

    it('can accept dropout', ({ expect }) => {
        const layer = new CausalSelfAttention(0, {
            biasInLayerNorm: false,
            vocabSize: 20,
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.1,
            blockSize: 4,
            mlpFactor: 4,
            useRope: true,
        });

        expect(layer).toBeInstanceOf(CausalSelfAttention);

        const input = tf.randomNormal([1, 4, 16]);
        const output = layer.call({ training: true }, input) as tf.Tensor;
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
        const attr = {
            training: false,
            attentionScores: { attentionOut: [] as tf.Tensor[] | undefined },
        };
        layer.call(attr, input);
        const attention = attr.attentionScores?.attentionOut;
        expect(attention).toHaveLength(1);
        expect(attention![0].shape).toEqual([2, 4, 4]);

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
            k: tf.randomNormal([1, 2, 4, 8]),
            v: tf.randomNormal([1, 2, 4, 8]),
            length: 2,
            cumulativeLength: 2,
        };
        const output = layer.call(
            {
                training: false,
                attentionScores: { attentionOut: undefined as tf.Tensor[] | undefined },
                pastKV: cache,
            },
            input
        ) as tf.Tensor;
        expect(output).toBeInstanceOf(tf.Tensor);
        expect(output.shape).toEqual([1, 1, 16]);
        const presentKV = cache;
        expect(presentKV).toBeDefined();
        expect(presentKV.k?.shape).toEqual([1, 2, 4, 8]);
        expect(presentKV.v?.shape).toEqual([1, 2, 4, 8]);
        expect(presentKV.length).toEqual(3);
        expect(presentKV.cumulativeLength).toEqual(3);
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
        layer.call({ training: false }, input); // Initialize the layer

        const weightsMap = new Map<string, tf.Tensor[]>();
        layer.saveWeights(weightsMap);

        const originalOutput = layer.call({ training: false }, input) as tf.Tensor;
        layer.dispose();

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
        newLayer.call({ training: false }, input); // Initialize the layer
        newLayer.loadWeights(weightsMap);

        const newOutput = newLayer.call({ training: false }, input) as tf.Tensor;
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
        const layer = new CausalSelfAttention(0, config);

        const input = tf.randomNormal([1, 4, 16]);
        const target = tf.randomNormal([1, 4, 16]);

        layer.call({ training: false, ropeCache }, input);

        //const optimizer = tf.train.adam(0.01);

        const f = () => {
            const output = layer.call({ training: true }, input) as tf.Tensor;
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
