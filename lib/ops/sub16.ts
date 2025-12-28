import { engine, sub, Tensor } from '@tensorflow/tfjs-core';

import { isPackedTensor } from '@base/utilities/packed';
// import './grads/sub16';

export function sub16(a: Tensor, b: Tensor): Tensor {
    if (!isPackedTensor(a) && !isPackedTensor(b)) {
        return sub(a, b);
    }
    const result = engine().runKernel('Sub16', { a, b }) as Tensor;
    return result;
}
