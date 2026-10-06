import { engine, Tensor } from '@tensorflow/tfjs-core';

//import './cpu/silu';
//import './webgl/silu';
import './grads/silu';

export function silu(x: Tensor): Tensor {
    return engine().runKernel('Silu', { x });
}

export function dSilu(dy: Tensor, x: Tensor): Tensor {
    return engine().runKernel('SiluGrad', { dy, x });
}
