import { gatherSub } from '../ops/gatherSub';
import { scatterSub } from '../ops/scatterSub';
import * as tf from '@tensorflow/tfjs-core';

/**
 * Numerically stable sparse cross-entropy with gradient support
 * This version handles potential numerical issues better
 */
export function sparseSoftmaxCrossEntropy(logits: tf.Tensor, labels: tf.Tensor, validMask?: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
        const numClasses = logits.shape[logits.shape.length - 1];
        const batchShape = logits.shape.slice(0, -1);
        const batchSize = batchShape.reduce((a, b) => a * b, 1);

        // Reshape logits to [batchSize, numClasses]
        const logits2d = logits.shape.length > 2 ? logits.reshape([batchSize, numClasses]) : logits;
        const labels1d = labels.shape.length > 1 ? labels.reshape([batchSize]).cast('int32') : labels.cast('int32');

        // TODO: Fuse max, sub, logSumExp and gatherSub

        // Subtract max for numerical stability
        const maxLogits = tf.max(logits2d, -1, true);
        const stableLogits = tf.sub(logits2d, maxLogits);

        const logSumExp = tf.logSumExp(stableLogits, -1);

        let loss = gatherSub(logSumExp, labels1d, stableLogits);
        if (validMask) {
            loss = tf.mul(loss, validMask);
            const validCount = tf.sum(validMask);
            loss = tf.div(tf.sum(loss), validCount);
        } else {
            loss = tf.mean(loss);
        }
        return loss;
    });
}

// TODO: Create custom operator.
export function createSoftmaxCrossEntropyWithGrad(masked?: boolean) {
    const ignoreIndex = -100;

    const sparseSoftmaxCrossEntropyGrad = tf.customGrad(
        // @ts-expect-error Invalid params
        (logits: tf.Tensor, labels: tf.Tensor, save: (tensor: tf.Tensor[]) => void) => {
            const numClasses = logits.shape[logits.shape.length - 1];
            const batchShape = logits.shape.slice(0, -1);
            const batchSize = batchShape.reduce((a, b) => a * b, 1);

            // Reshape logits and labels for computation
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

            const loss = sparseSoftmaxCrossEntropy(logits2d, safeLabels, validMask || undefined);
            save(validMask ? [logits2d, safeLabels, validMask] : [logits2d, safeLabels]);
            logits2d.dispose();
            labels1d.dispose();

            const grad = (dy: tf.Tensor, saved: tf.NamedTensorMap) => {
                return tf.tidy(() => {
                    const logitsSaved = saved[0];
                    const labelsSaved = saved[1];
                    const validMaskSaved = masked ? saved[2] : undefined;
                    //TODO: Fuse softmax and scatterSub
                    const softmaxProbs = tf.softmax(logitsSaved);

                    const count = validMaskSaved ? tf.sum(validMaskSaved) : tf.scalar(logitsSaved.shape[0], 'float32');

                    const dyScaled = dy.div(count);
                    const dyVec = dyScaled.broadcastTo([logitsSaved.shape[0]]);
                    const dyMasked = validMaskSaved && masked ? tf.mul(dyVec, validMaskSaved) : dyVec;

                    const gradLogitsScaled = scatterSub(softmaxProbs, labelsSaved, dyMasked);

                    // Gradient for labels is always zero
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
