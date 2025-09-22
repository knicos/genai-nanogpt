import { engine, Tensor } from '@tensorflow/tfjs-core';

import './cpu/gelu';
import './webgl/gelu';
import './grads/gelu';

export function gelu(x: Tensor): Tensor {
    return engine().runKernel('Gelu', { x });
}

export function dGelu(dy: Tensor, x: Tensor): Tensor {
    return engine().runKernel('GeluGrad', { dy, x });
}
