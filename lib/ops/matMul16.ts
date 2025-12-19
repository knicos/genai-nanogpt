import { Tensor, engine } from '@tensorflow/tfjs-core';

import './grads/matMul16';
import './cpu/matMul16';
import { isPackedTensor, packTensor } from '@base/utilities/packed';

export function matMul16(A: Tensor, B: Tensor, transposeA = false, transposeB = false): Tensor {
    const packed = isPackedTensor(A) && isPackedTensor(B);
    const t = engine().runKernel('MatMul16', { A, B }, { transposeA, transposeB }) as Tensor;
    return packed ? packTensor(t) : t;
}

export function matMul16Scaled(A: Tensor, B: Tensor, scale: number, transposeA = false, transposeB = false): Tensor {
    const packed = isPackedTensor(A) && isPackedTensor(B);
    const t = engine().runKernel('MatMul16', { A, B }, { transposeA, transposeB, scale }) as Tensor;
    return packed ? packTensor(t) : t;
}
