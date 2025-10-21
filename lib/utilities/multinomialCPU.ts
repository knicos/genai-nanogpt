import { tensor2d, Tensor2D } from '@tensorflow/tfjs-core';

export default function multinomialCPU(probs: number[]): Tensor2D {
    let cdf = 0;
    const rand = Math.random();
    for (let i = 0; i < probs.length; i++) {
        cdf += probs[i];
        if (rand < cdf) {
            return tensor2d([[i]], [1, 1], 'int32');
        }
    }

    return tensor2d([[probs.length - 1]], [1, 1], 'int32');
}
