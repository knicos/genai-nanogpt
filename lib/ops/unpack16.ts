import { engine, Tensor } from '@tensorflow/tfjs-core';

import './grads/unpack16';
import { isPackedTensor } from '@base/utilities/packed';

export function unpack16(x: Tensor, scaling = 1, disposeArg = false): Tensor {
    if (!isPackedTensor(x)) {
        return x;
    }
    const result = engine().runKernel('Unpack16', { x }, { scaling }) as Tensor;
    if (disposeArg) {
        x.dispose();
    }
    return result;
}
