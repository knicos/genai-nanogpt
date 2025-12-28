import { engine, mul, Tensor } from '@tensorflow/tfjs-core';

import { isPackedTensor } from '@base/utilities/packed';
// import './grads/mul16';

export function mul16(a: Tensor, b: Tensor): Tensor {
    if (!isPackedTensor(a) && !isPackedTensor(b)) {
        return mul(a, b);
    }
    const result = engine().runKernel('Mul16', { a, b }) as Tensor;
    return result;
}
