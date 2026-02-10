import { engine, Tensor } from '@tensorflow/tfjs-core';

import './cpu/matMulMul';
import './webgl/matMulMul';

export function matMulMul(
    x: Tensor,
    kernel: Tensor,
    y: Tensor,
    transposeA = false,
    transposeB = false
): Tensor {
    return engine().runKernel('MatMulMul', { x, kernel, y }, { transposeA, transposeB });
}
