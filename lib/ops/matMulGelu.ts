import { engine, Tensor } from '@tensorflow/tfjs-core';

import './cpu/matMulGelu';
import './webgl/matMulGelu';
import './grads/matMulGelu';

export function matMulGelu(x: Tensor, kernel: Tensor): Tensor {
    return engine().runKernel('MatMulGelu', { x, kernel });
}

export function dMatMulGelu(dy: Tensor, x: Tensor, kernel: Tensor): Tensor {
    return engine().runKernel('MatMulGeluGrad', { dy, x, kernel });
}
