import { Scalar, Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/adamMoments';
import './webgl/adamMoments';

export function adamMoments(moments: Tensor, gradient: Tensor, beta1: number, beta2: number, scaling: Scalar): Tensor {
    return engine().runKernel('AdamMoments', { moments, gradient, scaling }, { beta1, beta2 });
}
