import '@base/patches/engine';
import { describe, it } from 'vitest';
import { qkvCPU } from './cpu/qkv';
import * as tf from '@tensorflow/tfjs';
import { qkv } from './qkv';

describe('qkvRope', () => {
    it('produces equivalent gradients', async ({ expect }) => {
        const kernel = tf.variable(tf.randomNormal([16, 48]), true);

        const input = tf.randomNormal([1, 4, 16]);
        const target = tf.randomNormal([1, 4, 16]);

        //const optimizer = tf.train.adam(0.01);

        const f = () => {
            const [q, k, v] = qkv(input, kernel, 2);
            const output = tf.add(q, tf.add(k, v)).reshape([1, 4, 16]);
            const loss = tf.losses.meanSquaredError(target, output);
            return loss as tf.Scalar;
        };
        const { grads } = tf.variableGrads(f);

        const f2 = () => {
            const [q, k, v] = qkvCPU({ inputs: { x: input, kernel }, attrs: { heads: 2 } }) as tf.Tensor[];
            const output = tf.add(q, tf.add(k, v)).reshape([1, 4, 16]);
            const loss = tf.losses.meanSquaredError(target, output);
            return loss as tf.Scalar;
        };
        const { grads: grads2 } = tf.variableGrads(f2);

        const gradValues = Object.values(grads).map((grad) => {
            return grad.mean().dataSync()[0];
        });
        const gradValues2 = Object.values(grads2).map((grad) => {
            return grad.mean().dataSync()[0];
        });

        expect(gradValues.length).toBe(gradValues2.length);
        for (let i = 0; i < gradValues.length; i++) {
            expect(gradValues[i]).toBeCloseTo(gradValues2[i], 5);
        }
    });
});
