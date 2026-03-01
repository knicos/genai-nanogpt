import { gatherSub } from '../ops/gatherSub';
import { scatterSub } from '../ops/scatterSub';
import * as tf from '@tensorflow/tfjs-core';

/**
 * Numerically stable sparse cross-entropy with gradient support
 * This version handles potential numerical issues better
 */
export function sparseSoftmaxCrossEntropy(
    logits: tf.Tensor,
    labels: tf.Tensor,
    validMask?: tf.Tensor,
    keepBatch?: boolean,
    originalBatchShape?: number[],
    labelSmoothing = 0
): tf.Tensor {
    return tf.tidy(() => {
        const numClasses = logits.shape[logits.shape.length - 1];
        const batchShape = originalBatchShape ? originalBatchShape : logits.shape.slice(0, -1);
        const batchSize = batchShape.reduce((a, b) => a * b, 1);

        const logits2d = logits.shape.length > 2 ? logits.reshape([batchSize, numClasses]) : logits;
        const labels1d = labels.shape.length > 1 ? labels.reshape([batchSize]).cast('int32') : labels.cast('int32');

        const maxLogits = tf.max(logits2d, -1, true);
        const stableLogits = tf.sub(logits2d, maxLogits);
        const logSumExp = tf.logSumExp(stableLogits, -1);

        const nllHard = gatherSub(logSumExp, labels1d, stableLogits);

        let loss: tf.Tensor;
        if (labelSmoothing > 0) {
            // Uniform-target cross-entropy: logSumExp - mean(logits)
            const nllUniform = tf.sub(logSumExp, tf.mean(stableLogits, -1));
            loss = tf.add(tf.mul(nllHard, 1 - labelSmoothing), tf.mul(nllUniform, labelSmoothing));
        } else {
            loss = nllHard;
        }

        if (validMask) {
            loss = tf.mul(loss, validMask);
            if (keepBatch) {
                const validCount = tf.sum(validMask.reshape(batchShape), -1);
                loss = tf.div(tf.sum(loss.reshape(batchShape), -1), validCount);
            } else {
                const validCount = tf.sum(validMask);
                loss = tf.div(tf.sum(loss), validCount);
            }
        } else {
            if (keepBatch) {
                loss = tf.mean(loss.reshape(batchShape), -1);
            } else {
                loss = tf.mean(loss);
            }
        }
        return loss;
    });
}

// TODO: Create custom operator.
export function createSoftmaxCrossEntropyWithGrad(masked?: boolean, keepBatch?: boolean, labelSmoothing = 0) {
    const ignoreIndex = -100;

    const sparseSoftmaxCrossEntropyGrad = tf.customGrad(
        // @ts-expect-error Invalid params
        (logits: tf.Tensor, labels: tf.Tensor, save: (tensor: tf.Tensor[]) => void) => {
            const numClasses = logits.shape[logits.shape.length - 1];
            const batchShape = logits.shape.slice(0, -1);
            const batchSize = batchShape.reduce((a, b) => a * b, 1);

            const logits2d = logits.reshape([batchSize, numClasses]);
            const labels1d = labels.reshape([batchSize]).cast('int32');

            let safeLabels: tf.Tensor;
            let validMask: tf.Tensor | null = null;
            if (masked) {
                const ignoreTensor = tf.scalar(ignoreIndex, 'int32');
                const validMaskBool = tf.notEqual(labels1d, ignoreTensor);
                validMask = validMaskBool.cast('float32');
                safeLabels = tf.where(validMaskBool, labels1d, tf.zerosLike(labels1d));
            } else {
                safeLabels = labels1d;
            }

            const loss = sparseSoftmaxCrossEntropy(
                logits2d,
                safeLabels,
                validMask || undefined,
                keepBatch,
                batchShape,
                labelSmoothing
            );

            save(validMask ? [logits2d, safeLabels, validMask] : [logits2d, safeLabels]);
            logits2d.dispose();
            labels1d.dispose();

            const grad = (dy: tf.Tensor, saved: tf.NamedTensorMap) => {
                return tf.tidy(() => {
                    const logitsSaved = saved[0];
                    const labelsSaved = saved[1];
                    const validMaskSaved = masked ? saved[2] : undefined;

                    const softmaxProbs = tf.softmax(logitsSaved);
                    const count = validMaskSaved ? tf.sum(validMaskSaved) : tf.scalar(logitsSaved.shape[0], 'float32');

                    const dyScaled = dy.div(count);
                    const dyVec = dyScaled.broadcastTo([logitsSaved.shape[0]]);
                    const dyMasked = validMaskSaved && masked ? tf.mul(dyVec, validMaskSaved) : dyVec;

                    let gradLogitsScaled: tf.Tensor;
                    if (labelSmoothing > 0) {
                        const uniform = labelSmoothing / numClasses;
                        const probsMinusUniform = tf.sub(softmaxProbs, uniform);
                        gradLogitsScaled = scatterSub(
                            probsMinusUniform,
                            labelsSaved,
                            tf.mul(dyMasked, 1 - labelSmoothing)
                        );
                    } else {
                        gradLogitsScaled = scatterSub(softmaxProbs, labelsSaved, dyMasked);
                    }

                    const gradLabels = tf.zerosLike(labels);
                    const reshapedGradLogits = gradLogitsScaled.reshape(logits.shape);

                    return [reshapedGradLogits, gradLabels];
                });
            };

            return { value: loss, gradFunc: grad };
        }
    );
    return sparseSoftmaxCrossEntropyGrad;
}
