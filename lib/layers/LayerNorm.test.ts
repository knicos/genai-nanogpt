import { afterEach, describe, it } from 'vitest';
import LayerNorm from './LayerNorm';
import * as tf from '@tensorflow/tfjs';

describe('LayerNorm', () => {
    afterEach(() => {
        tf.disposeVariables();
    });

    it('should normalize the input tensor', ({ expect }) => {
        const input = tf.tensor([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        const layerNorm = new LayerNorm(tf, [3], 1e-5, 'layer_norm');
        const output = layerNorm.apply(input);

        expect(output.shape).toEqual([2, 3]);
        const mean = output.mean(-1, true);
        const variance = output.sub(mean).square().mean(-1);
        expect(mean.dataSync()[0]).toBeCloseTo(0);
        expect(variance.dataSync()[0]).toBeCloseTo(1);
    });

    it('should scale with gamma', ({ expect }) => {
        const input = tf.tensor([
            [1, 2, 3],
            [4, 5, 6],
        ]);
        const layerNorm = new LayerNorm(tf, [3], 1e-5, 'layer_norm');
        layerNorm.setWeights([tf.tensor1d([2, 2, 2])]); // Set gamma to 2
        const output = layerNorm.apply(input);

        expect(output.shape).toEqual([2, 3]);
        const mean = output.mean(-1, true);
        const variance = output.sub(mean).square().mean(-1);
        expect(mean.dataSync()[0]).toBeCloseTo(0);
        expect(variance.dataSync()[0]).toBeCloseTo(4);
    });
});
