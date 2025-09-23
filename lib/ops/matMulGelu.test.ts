import { describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { matMulGelu } from './matMulGelu';
import { gelu } from './gelu';

describe('matMulGelu', () => {
    it('produces equivalent gradients', async ({ expect }) => {
        const batch = 2;
        const inDim = 8;
        const outDim = 16;

        const kernel = tf.variable(tf.randomNormal([inDim, outDim]), true);

        const input = tf.randomNormal([batch, inDim]);
        const target = tf.randomNormal([batch, outDim]);

        //const optimizer = tf.train.adam(0.01);

        const f = () => {
            const h = matMulGelu(input, kernel);
            const loss = tf.losses.meanSquaredError(target, h);
            return loss as tf.Scalar;
        };
        const { grads } = tf.variableGrads(f);

        const f2 = () => {
            const m = tf.matMul(input, kernel);
            const h = gelu(m);
            const loss = tf.losses.meanSquaredError(target, h);
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
