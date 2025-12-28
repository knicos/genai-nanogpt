import { add, engine, Tensor } from '@tensorflow/tfjs-core';

import { isPackedTensor } from '@base/utilities/packed';
import './grads/add16';

export function add16(a: Tensor, b: Tensor): Tensor {
    if (!isPackedTensor(a) && !isPackedTensor(b)) {
        return add(a, b);
    }
    const result = engine().runKernel('Add16', { a, b }) as Tensor;
    return result;
}
