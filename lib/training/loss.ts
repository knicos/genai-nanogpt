import { Tensor } from '@tensorflow/tfjs-core';
import { createSoftmaxCrossEntropyWithGrad } from './sparseCrossEntropy';

export function calculateLoss(logits: Tensor, targets: Tensor, masked?: boolean): Tensor {
    try {
        //return this.tf.losses.softmaxCrossEntropy(targets, logits, this.tf.Reduction.MEAN);
        const lossFn = createSoftmaxCrossEntropyWithGrad(masked);
        const loss = lossFn(logits, targets);
        return loss;
    } catch (error) {
        console.error('Error computing loss:', error);
        throw new Error(`Loss computation failed: ${error}`);
    }
}
