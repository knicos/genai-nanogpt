import { describe, it } from 'vitest';
import { importWeights, exportWeights } from './weights';
import * as tf from '@tensorflow/tfjs';

describe('importWeights', () => {
    it('should import weights correctly', async ({ expect }) => {
        const manifest = {
            spec: [
                { shape: [2, 2], min: 0, scale: 1 },
                { shape: [3, 3], min: 0, scale: 1 },
            ],
            data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
        };

        const weights = await importWeights(manifest, tf);

        expect(weights.length).toBe(2);
        expect(weights[0].shape).toEqual([2, 2]);
        expect(weights[1].shape).toEqual([3, 3]);
        expect(weights[0].dataSync()).toEqual(new Float32Array([1, 2, 3, 4]));
        expect(weights[1].dataSync()).toEqual(new Float32Array([5, 6, 7, 8, 9, 10, 11, 12, 13]));
    });

    it('can import previously exported weights', async ({ expect }) => {
        const weights = [
            tf.tensor2d([
                [1, 2],
                [3, 4],
            ]),
            tf.tensor2d([
                [5, 6, 7],
                [8, 9, 10],
            ]),
        ];
        const manifest = await exportWeights(weights);
        const importedWeights = await importWeights(manifest, tf);

        expect(importedWeights.length).toBe(2);
        expect(importedWeights[0].shape).toEqual([2, 2]);
        expect(importedWeights[1].shape).toEqual([2, 3]);
        expect(importedWeights[0].dataSync()).toEqual(new Float32Array([1, 2, 3, 4]));
        expect(importedWeights[1].dataSync()).toEqual(new Float32Array([5, 6, 7, 8, 9, 10]));
    });
});

describe('exportWeights', () => {
    it('should export weights correctly', async ({ expect }) => {
        const weights = [
            tf.tensor2d([
                [1, 2],
                [3, 4],
            ]),
            tf.tensor2d([
                [5, 6, 7],
                [8, 9, 10],
            ]),
        ];
        const manifest = await exportWeights(weights);

        expect(manifest.spec.length).toBe(2);
        expect(manifest.spec[0].shape).toEqual([2, 2]);
        expect(manifest.spec[1].shape).toEqual([2, 3]);
        expect(manifest.data.length).toBe(10);
        expect(manifest.data[0]).toBe(1);
        expect(manifest.data[1]).toBe(2);
    });
});
