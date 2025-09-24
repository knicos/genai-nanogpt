import { describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { normRMS } from './normRMS';

function arraysClose(a: unknown, b: unknown, epsilon = 1e-5) {
    if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; ++i) {
            if (!arraysClose(a[i], b[i], epsilon)) return false;
        }
        return true;
    } else if (typeof a === 'number' && typeof b === 'number') {
        if (a === -Infinity && b === -Infinity) return true;
        return Math.abs(a - b) < epsilon;
    } else {
        return false;
    }
}

describe('normRMS', () => {
    it('produces equivalent gradients', async ({ expect }) => {
        const batch = 2;
        const seqLen = 8;
        const channels = 16;

        const gamma = tf.variable(tf.randomNormal([channels]), true);
        const input = tf.randomNormal([batch, seqLen, channels]);
        const target = tf.randomNormal([batch, seqLen, channels]);

        //const optimizer = tf.train.adam(0.01);

        const f = () => {
            const h = normRMS(input, gamma);
            const loss = tf.losses.meanSquaredError(target, h);
            return loss as tf.Scalar;
        };
        const { grads } = tf.variableGrads(f);

        const f2 = () => {
            const meanSquare = input.square().mean(-1, true);
            const invRms = meanSquare.add(1e-8).rsqrt();
            const normalized = input.mul(invRms);
            const result = normalized.mul(gamma);
            const loss = tf.losses.meanSquaredError(target, result);
            return loss as tf.Scalar;
        };
        const { grads: grads2 } = tf.variableGrads(f2);

        const gradValues = Object.values(grads).map((grad) => {
            return grad.arraySync();
        });
        const gradValues2 = Object.values(grads2).map((grad) => {
            return grad.arraySync();
        });

        expect(arraysClose(gradValues, gradValues2, 1e-5)).toBe(true);
    });
});
