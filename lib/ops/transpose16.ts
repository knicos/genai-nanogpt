import {
    engine,
    GradConfig,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerGradient,
    registerKernel,
    Tensor,
    TensorInfo,
    transpose,
    TransposeAttrs,
} from '@tensorflow/tfjs-core';
import { forceFloat, forcePacked } from './grads/utils';
import { getUndoAxesPermutation } from '@tensorflow/tfjs-core/dist/ops/axis_util';
import { isPackedTensor } from '@base/utilities/packed';

export const transpose16GradConfig: GradConfig = {
    kernelName: 'Transpose16',
    gradFunc: (dy: Tensor | Tensor[], _: Tensor[], attrs: NamedAttrMap) => {
        if (Array.isArray(dy)) {
            throw new Error('Transpose16 gradient does not support multiple outputs.');
        }

        const transposeAttrs: TransposeAttrs = attrs as unknown as TransposeAttrs;
        const { perm } = transposeAttrs;
        const undoPerm = getUndoAxesPermutation(perm);
        return { x: () => transpose16(dy, undoPerm) };
    },
};

registerGradient(transpose16GradConfig);

function transpose16_(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { inputs, attrs } = args;
    const { x } = inputs as { x: Tensor };
    const { perm } = attrs as unknown as TransposeAttrs;

    const packed = isPackedTensor(x);

    if (packed && perm[perm.length - 1] !== x.shape.length - 1) {
        throw new Error('Transpose16 currently only supports the last axis being unchanged.');
    }

    const result = packed ? forcePacked(transpose(forceFloat(x) as Tensor, perm)) : transpose(x, perm);
    return result;
}

const webglConfig: KernelConfig = {
    kernelName: 'Transpose16',
    backendName: 'webgl',
    kernelFunc: transpose16_,
};

registerKernel(webglConfig);

const cpuConfig: KernelConfig = {
    kernelName: 'Transpose16',
    backendName: 'cpu',
    kernelFunc: transpose16_,
};

registerKernel(cpuConfig);

export function transpose16(x: Tensor, perm?: number[]): Tensor {
    if (perm == null) {
        perm = x.shape.map((_, i) => i).reverse();
    }
    // Ideally check this but might result in incorrect dispose.
    /*if (perm.every((p, i) => p === i)) {
        return x;
    }*/
    const r = engine().runKernel('Transpose16', { x }, { perm }) as Tensor;
    // packTensor(r);
    return r;
}
