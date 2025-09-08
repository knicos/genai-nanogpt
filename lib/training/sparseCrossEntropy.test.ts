import { describe, it } from 'vitest';
import { createSoftmaxCrossEntropyWithGrad, sparseSoftmaxCrossEntropy } from './sparseCrossEntropy';
import * as tf from '@tensorflow/tfjs';

describe('sparseCrossEntropy', () => {
    it('should compute loss correctly', ({ expect }) => {
        const logits = tf.tensor2d(
            [
                [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [2, 13]
        );
        const labels = tf.tensor1d([2, 1], 'int32');

        const loss = tf.tidy(() => {
            const r = sparseSoftmaxCrossEntropy(logits, labels).mean();
            console.log('SPARSE', tf.memory());
            return r;
        });

        const lossValue = loss.arraySync();
        loss.dispose();

        const comparison = tf.tidy(() => {
            const r = tf.losses.softmaxCrossEntropy(tf.oneHot(labels, 13), logits);
            console.log('COMPARISON', tf.memory());
            return r;
        });

        // Expect the comparison to be the mean of the loss, approximately
        expect(lossValue).toBeCloseTo(comparison.arraySync() as number, 5);

        logits.dispose();
        labels.dispose();
        comparison.dispose();
    });

    it('produces correct gradients', ({ expect }) => {
        const logits = tf.tensor2d(
            [
                [1, 2, 3, 0],
                [1, 3, 2, 0],
            ],
            [2, 4]
        );
        const labels = tf.tensor1d([2, 1], 'int32');

        const lossFun = createSoftmaxCrossEntropyWithGrad();
        const f = (x: tf.Tensor) => lossFun(x as tf.Tensor2D, labels).mean() as tf.Scalar;

        const { value, grad } = tf.valueAndGrad(f)(logits);

        const comparison = tf.tidy(() => {
            const f = (x: tf.Tensor) => tf.losses.softmaxCrossEntropy(tf.oneHot(labels, 4), x);
            return tf.valueAndGrad(f)(logits);
        });

        const gradValue = grad.arraySync() as number[][];
        const comparisonValue = comparison.grad.arraySync() as number[][];

        expect(gradValue[0][0]).toBeCloseTo(comparisonValue[0][0], 5);
        expect(gradValue[0][1]).toBeCloseTo(comparisonValue[0][1], 5);
        expect(gradValue[0][2]).toBeCloseTo(comparisonValue[0][2], 5);
        expect(gradValue[0][3]).toBeCloseTo(comparisonValue[0][3], 5);
        expect(gradValue[1][0]).toBeCloseTo(comparisonValue[1][0], 5);
        expect(gradValue[1][1]).toBeCloseTo(comparisonValue[1][1], 5);
        expect(gradValue[1][2]).toBeCloseTo(comparisonValue[1][2], 5);
        expect(gradValue[1][3]).toBeCloseTo(comparisonValue[1][3], 5);

        comparison.grad.dispose();
        comparison.value.dispose();
        logits.dispose();
        labels.dispose();
        value.dispose();
        grad.dispose();
    });
});
