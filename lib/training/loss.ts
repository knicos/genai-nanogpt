import { Tensor } from '@tensorflow/tfjs-core';
import { createSoftmaxCrossEntropyWithGrad } from './sparseCrossEntropy';

export function calculateLoss(
    logits: Tensor,
    targets: Tensor,
    masked?: boolean,
    keepBatch?: boolean,
    labelSmoothing?: number
): Tensor {
    try {
        //return this.tf.losses.softmaxCrossEntropy(targets, logits, this.tf.Reduction.MEAN);
        const lossFn = createSoftmaxCrossEntropyWithGrad(
            masked,
            keepBatch,
            labelSmoothing && labelSmoothing > 0 ? labelSmoothing : undefined
        );
        const loss = lossFn(logits, targets);
        return loss;
    } catch (error) {
        console.error('Error computing loss:', error);
        throw new Error(`Loss computation failed: ${error}`);
    }
}

export function calculateAccuracy(logits: Tensor, targets: Tensor): Tensor {
    try {
        const predictions = logits.argMax(-1);
        const correct = predictions.equal(targets).cast('float32');
        const accuracy = correct.mean();
        predictions.dispose();
        correct.dispose();
        return accuracy;
    } catch (error) {
        console.error('Error computing accuracy:', error);
        throw new Error(`Accuracy computation failed: ${error}`);
    }
}
