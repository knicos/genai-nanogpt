import { engine, Tensor } from '@tensorflow/tfjs-core';

import './cpu/normRMS';
import './webgl/normRMS';
import './grads/normRMS';

export function normRMS(x: Tensor, gamma?: Tensor): Tensor {
    return engine().runKernel(gamma ? 'RMSNorm' : 'RMSNormNoGamma', gamma ? { x, gamma } : { x });
}
