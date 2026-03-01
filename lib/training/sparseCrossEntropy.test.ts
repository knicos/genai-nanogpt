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
            const r = sparseSoftmaxCrossEntropy(logits, labels);
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

    it('can compute loss with keepBatch option', ({ expect }) => {
        const logits = tf.tensor3d(
            [
                [
                    [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ],
            [2, 2, 13]
        );
        const labels = tf.tensor2d(
            [
                [2, 1],
                [2, 1],
            ],
            [2, 2],
            'int32'
        );

        const loss = tf.tidy(() => {
            const r = sparseSoftmaxCrossEntropy(logits, labels, undefined, true);
            return r;
        });

        const lossValue = loss.arraySync();
        loss.dispose();

        console.log('Loss value with keepBatch:', lossValue);

        expect(lossValue).toHaveLength(2);

        logits.dispose();
        labels.dispose();
    });

    it('should compute masked loss correctly', ({ expect }) => {
        const logits = tf.tensor2d(
            [
                [1, 3, 2, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 3, 2, 0, 0, 0, 7, 0, 0, 6, 0, 0, 0],
            ],
            [7, 13]
        );
        const labels = tf.tensor1d([-100, 2, 3, 1, 5, -100, -100], 'int32');

        const lossFn = createSoftmaxCrossEntropyWithGrad(true);

        const loss = tf.tidy(() => {
            const r = lossFn(logits, labels);
            return r;
        });

        const lossValue = loss.arraySync();
        loss.dispose();

        expect(lossValue).toBeCloseTo(0.0, 5);
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
        const f = (x: tf.Tensor) => lossFun(x as tf.Tensor2D, labels) as tf.Scalar;

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

    it('produces correct masked gradients', ({ expect }) => {
        const logits = tf.tensor2d(
            [
                [1, 2, 3, 0],
                [1, 3, 2, 0],
            ],
            [2, 4]
        );
        const labels = tf.tensor1d([2, -100], 'int32');

        const lossFun = createSoftmaxCrossEntropyWithGrad(true);
        const f = (x: tf.Tensor) => lossFun(x as tf.Tensor2D, labels) as tf.Scalar;

        const { value, grad } = tf.valueAndGrad(f)(logits);

        const gradValue = grad.arraySync() as number[][];

        expect(gradValue[1][0]).toBeCloseTo(0.0, 5);
        expect(gradValue[1][1]).toBeCloseTo(0.0, 5);
        expect(gradValue[1][2]).toBeCloseTo(0.0, 5);
        expect(gradValue[1][3]).toBeCloseTo(0.0, 5);

        logits.dispose();
        labels.dispose();
        value.dispose();
        grad.dispose();
    });

    it('produces correct masked gradients (matches reference with multiple valid tokens)', ({ expect }) => {
        const logits = tf.tensor2d(
            [
                [1, 2, 3, 0], // valid (label 2)
                [2, 1, 0, 3], // valid (label 3)
                [1, 3, 2, 0], // masked
            ],
            [3, 4]
        );
        const labels = tf.tensor1d([2, 3, -100], 'int32');

        const lossFun = createSoftmaxCrossEntropyWithGrad(true);
        const fCustom = (x: tf.Tensor) => lossFun(x as tf.Tensor2D, labels) as tf.Scalar;
        const custom = tf.valueAndGrad(fCustom)(logits);

        // Reference loss: mean over valid tokens only
        const fRef = (x: tf.Tensor) =>
            tf.tidy(() => {
                const x2d = x as tf.Tensor2D;
                const validMaskBool = tf.notEqual(labels, tf.scalar(-100, 'int32'));
                const validMask = validMaskBool.cast('float32');
                const safeLabels = tf.where(validMaskBool, labels, tf.zerosLike(labels));

                const logProbs = tf.logSoftmax(x2d, -1); // [N, C]
                const oneHot = tf.oneHot(safeLabels, 4); // [N, C]
                const perTokenLoss = tf.neg(tf.sum(tf.mul(oneHot, logProbs), -1)); // [N]

                const maskedLoss = tf.mul(perTokenLoss, validMask);
                const denom = tf.maximum(tf.sum(validMask), tf.scalar(1, 'float32'));

                return tf.div(tf.sum(maskedLoss), denom) as tf.Scalar;
            });

        const reference = tf.valueAndGrad(fRef)(logits);

        const customGrad = custom.grad.arraySync() as number[][];
        const refGrad = reference.grad.arraySync() as number[][];

        // Full gradient comparison detects scaling errors on non-target classes
        for (let i = 0; i < customGrad.length; i++) {
            for (let j = 0; j < customGrad[i].length; j++) {
                expect(customGrad[i][j]).toBeCloseTo(refGrad[i][j], 5);
            }
        }

        // Masked row should be exactly zero-gradient
        expect(customGrad[2][0]).toBeCloseTo(0.0, 6);
        expect(customGrad[2][1]).toBeCloseTo(0.0, 6);
        expect(customGrad[2][2]).toBeCloseTo(0.0, 6);
        expect(customGrad[2][3]).toBeCloseTo(0.0, 6);

        logits.dispose();
        labels.dispose();
        custom.value.dispose();
        custom.grad.dispose();
        reference.value.dispose();
        reference.grad.dispose();
    });
});
