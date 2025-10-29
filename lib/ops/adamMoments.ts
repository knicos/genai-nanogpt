import { Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/adamMoments';
import './webgl/adamMoments';

export function adamMoments(moments: Tensor, gradient: Tensor, beta1: number, beta2: number): Tensor {
    return engine().runKernel('AdamMoments', { moments, gradient }, { beta1, beta2 });
}
