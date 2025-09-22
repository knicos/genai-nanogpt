import { describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { appendCache } from './appendCache';

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

describe('appendCache', () => {
    it('fills a middle position', async ({ expect }) => {
        const cache = tf.tensor4d(
            [
                [
                    [
                        [0.1, 0.2, 0, 0],
                        [0.1, 0.2, 0, 0],
                        [0.0, 0.0, 0, 0],
                        [0.0, 0.0, 0, 0],
                    ],
                ],
            ],
            [1, 1, 4, 4]
        );
        const x = tf.tensor4d([[[[0.1, 0.2, 0.3, 0.4]]]], [1, 1, 1, 4]);

        const r = appendCache(x, 4, 2, cache);
        const rArray = await r.array();
        expect(
            arraysClose(rArray, [
                [
                    [
                        [0.1, 0.2, 0, 0],
                        [0.1, 0.2, 0, 0],
                        [0.1, 0.2, 0.3, 0.4],
                        [0.0, 0.0, 0, 0],
                    ],
                ],
            ])
        ).toBe(true);
    });

    it('works with an empty cache', async ({ expect }) => {
        const x = tf.tensor4d([[[[0.1, 0.2, 0.3, 0.4]]]], [1, 1, 1, 4]);

        const r = appendCache(x, 4, 0);
        const rArray = await r.array();
        expect(
            arraysClose(rArray, [
                [
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                ],
            ])
        ).toBe(true);
    });

    it('shifts and pushes a new item', async ({ expect }) => {
        const cache = tf.tensor4d(
            [
                [
                    [
                        [0.1, 0.2, 0, 0],
                        [0.1, 0.2, 0, 0],
                        [0.0, 0.0, 0, 0],
                        [0.0, 0.0, 0, 0],
                    ],
                ],
            ],
            [1, 1, 4, 4]
        );
        const x = tf.tensor4d([[[[0.1, 0.2, 0.3, 0.4]]]], [1, 1, 1, 4]);

        const r = appendCache(x, 4, 4, cache);
        const rArray = await r.array();
        expect(
            arraysClose(rArray, [
                [
                    [
                        [0.1, 0.2, 0, 0],
                        [0.0, 0.0, 0, 0],
                        [0.0, 0.0, 0, 0],
                        [0.1, 0.2, 0.3, 0.4],
                    ],
                ],
            ])
        ).toBe(true);
    });
});
