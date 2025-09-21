import { Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/fusedSoftmax';
import './webgl/fusedSoftmax';
import './grads/fusedSoftmax';

export function fusedSoftmax(logits: Tensor, dropoutRate: number, seed: number): Tensor {
    return engine().runKernel('FusedSoftmax', { logits }, { dim: -1, dropoutRate, seed });
}
