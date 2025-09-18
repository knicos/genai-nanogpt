import { describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { rope } from './rope';
import RoPECache from '@base/layers/RoPECache';
import { ropeCPU } from './cpu/rope';

describe('rope', () => {
    it('produces equivalent gradients', async ({ expect }) => {
        const cache = new RoPECache({
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

        const input = tf.randomNormal([1, 2, 4, 8]);

        cache.ensureRopeCache(16);

        //const optimizer = tf.train.adam(0.01);

        const f = () => {
            const r = rope(input, cache, 0);
            return r;
        };
        const grads = tf.grad(f)(input);

        const f2 = () => {
            const r = ropeCPU({
                inputs: { x: input, cos: cache.getCos()!, sin: cache.getSin()! },
                attrs: { pastLen: 0 },
            }) as tf.Tensor;
            return r;
        };
        const grads2 = tf.grad(f2)(input);

        expect(grads.shape).toEqual(grads2.shape);
        const fusedMean = grads.mean().dataSync()[0];
        const unfusedMean = grads2.mean().dataSync()[0];
        expect(fusedMean).toBeCloseTo(unfusedMean, 5);
    });
});
