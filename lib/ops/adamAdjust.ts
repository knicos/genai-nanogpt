import { Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/adamAdjust';
import './webgl/adamAdjust';

export function adamAdjust(
    moments: Tensor,
    value: Tensor,
    beta1: number,
    beta2: number,
    epsilon: number,
    learningRate: number
): Tensor {
    return engine().runKernel('AdamAdjust', { moments, value }, { beta1, beta2, epsilon, learningRate });
}
