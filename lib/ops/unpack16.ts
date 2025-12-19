import { engine, Tensor } from '@tensorflow/tfjs-core';

import './grads/unpack16';
import { isPackedTensor } from '@base/utilities/packed';

export function unpack16(x: Tensor, scaling = 1): Tensor {
    if (!isPackedTensor(x)) {
        return x;
    }
    return engine().runKernel('Unpack16', { x }, { scaling });
}
