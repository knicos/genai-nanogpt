import { Tensor, engine } from '@tensorflow/tfjs-core';

import './grads/softmax16';
import { isPackedTensor } from '@base/utilities/packed';

export function softmax16(logits: Tensor): Tensor {
    if (!isPackedTensor(logits)) {
        return engine().runKernel('Softmax', { logits }, { dim: logits.rank - 1 }) as Tensor;
    }
    const r = engine().runKernel('Softmax16', { logits }, { dim: logits.rank - 1 }) as Tensor;
    return r;
}
