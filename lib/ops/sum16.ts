import { engine, sum, Tensor } from '@tensorflow/tfjs-core';

import { isPackedTensor } from '@base/utilities/packed';

export function sum16(x: Tensor, axis?: number | number[], keepDims = false): Tensor {
    if (!isPackedTensor(x)) {
        return sum(x, axis, keepDims);
    }
    if (keepDims) {
        throw new Error('sum16 with keepDims=true not supported for packed tensors');
    }
    const result = engine().runKernel('Sum16', { x }, { axis: axis ?? -1, keepDims }) as Tensor;
    return result;
}
