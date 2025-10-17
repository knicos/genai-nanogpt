import { gatherSub } from '../ops/gatherSub';
import { scatterSub } from '../ops/scatterSub';
import * as tf from '@tensorflow/tfjs-core';

/**
 * Numerically stable sparse cross-entropy with gradient support
 * This version handles potential numerical issues better
 */
export function sparseSoftmaxCrossEntropy(logits: tf.Tensor, labels: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
        const numClasses = logits.shape[logits.shape.length - 1];
        const batchShape = logits.shape.slice(0, -1);
        const batchSize = batchShape.reduce((a, b) => a * b, 1);

        // Reshape logits to [batchSize, numClasses]
        const logits2d = logits.shape.length > 2 ? logits.reshape([batchSize, numClasses]) : logits;
        const labels1d = labels.shape.length > 1 ? labels.reshape([batchSize]).cast('int32') : labels.cast('int32');

        // Subtract max for numerical stability
        const maxLogits = tf.max(logits2d, -1, true);
        const stableLogits = tf.sub(logits2d, maxLogits);

        const logSumExp = tf.logSumExp(stableLogits, -1);

        const loss = gatherSub(logSumExp, labels1d, stableLogits);
        return loss;
    });
}

/**
 * Custom gradient implementation for sparse cross-entropy
 * This ensures proper backpropagation
 */
export function createSoftmaxCrossEntropyWithGrad() {
    const sparseSoftmaxCrossEntropyGrad = tf.customGrad(
        // @ts-expect-error Invalid params
        (logits: tf.Tensor, labels: tf.Tensor, save: (tensor: tf.Tensor[]) => void) => {
            const numClasses = logits.shape[logits.shape.length - 1];
            const batchShape = logits.shape.slice(0, -1);
            const batchSize = batchShape.reduce((a, b) => a * b, 1);

            // Reshape logits and labels for computation
            const logits2d = logits.reshape([batchSize, numClasses]);
            const labels1d = labels.reshape([batchSize]).cast('int32');
            const loss = sparseSoftmaxCrossEntropy(logits2d, labels1d);
            save([logits2d, labels1d]);
            logits2d.dispose();
            labels1d.dispose();

            const grad = (dy: tf.Tensor, saved: tf.NamedTensorMap) => {
                return tf.tidy(() => {
                    const logitsSaved = saved[0];
                    const labelsSaved = saved[1];
                    const softmaxProbs = tf.softmax(logitsSaved);

                    const gradLogitsScaled = scatterSub(softmaxProbs, labelsSaved, dy);

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
