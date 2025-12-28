import { Tensor, engine } from '@tensorflow/tfjs-core';

import './grads/matMul16';
import './webgl/matMul16';
import './cpu/matMul16';
import { isPackedTensor, packTensor } from '@base/utilities/packed';
import { pack16 } from './pack16';

export function matMul16(
    A: Tensor,
    B: Tensor,
    transposeA = false,
    transposeB = false,
    attrs: {
        scale?: number;
        scaleA?: number;
        scaleB?: number;
        activation?: 'gelu';
        forceOutputShape?: number[];
        perm?: number[];
    } = {}
): Tensor {
    const isPackedA = isPackedTensor(A);
    const isPackedB = isPackedTensor(B);
    const packed = isPackedA || isPackedB;

    const pA = !packed || isPackedA ? A : pack16(A);
    const pB = !packed || isPackedB ? B : pack16(B);

    const t = engine().runKernel('MatMul16', { A: pA, B: pB }, { transposeA, transposeB, ...attrs }) as Tensor;

    if (packed && !isPackedA) {
        pA.dispose();
    }
    if (packed && !isPackedB) {
        pB.dispose();
    }

    return packed ? packTensor(t) : t;
}

export function matMul16Scaled(A: Tensor, B: Tensor, scale: number, transposeA = false, transposeB = false): Tensor {
    return matMul16(A, B, transposeA, transposeB, { scale });
}

export function matMul16ScaleA(A: Tensor, B: Tensor, scale: number, transposeA = false, transposeB = false): Tensor {
    return matMul16(A, B, transposeA, transposeB, { scaleA: scale });
}

export function matMul16ScaleB(A: Tensor, B: Tensor, scale: number, transposeA = false, transposeB = false): Tensor {
    return matMul16(A, B, transposeA, transposeB, { scaleB: scale });
}

export function matMul16Gelu(A: Tensor, B: Tensor, transposeA = false, transposeB = false): Tensor {
    return matMul16(A, B, transposeA, transposeB, { activation: 'gelu' });
}
