import { describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { fusedSoftmax } from './fusedSoftmax';

describe('fusedSoftmax', () => {
    it('produces equivalent gradients', async ({ expect }) => {
        const x = tf.tensor4d(
            [
                [
                    [
                        [0.1, 0.2],
                        [0.3, 0.4],
                    ],
                    [
                        [0.1, 0.2],
                        [0.3, 0.4],
                    ],
                ],
            ],
            [1, 2, 2, 2]
        ); // (1,nh,T,T)

        const f = () => {
            const r = fusedSoftmax(x, 0, 0);
            return r;
        };
        const grads = tf.grad(f)(x);

        const f2 = () => {
            const r = tf.softmax(x, -1);
            return r;
        };
        const grads2 = tf.grad(f2)(x);

        expect(grads.shape).toEqual(grads2.shape);
        const fusedMean = grads.mean().dataSync()[0];
        const unfusedMean = grads2.mean().dataSync()[0];
        expect(fusedMean).toBeCloseTo(unfusedMean, 5);
    });
});
