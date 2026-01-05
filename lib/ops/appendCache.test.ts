import '@base/patches/engine';
import { describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { appendCache } from './appendCache';
import { arraysClose } from '@base/utilities/arrayClose';

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
        ).toBeLessThan(1e-6);
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
        ).toBeLessThan(1e-6);
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
        ).toBeLessThan(1e-6);
    });
});
