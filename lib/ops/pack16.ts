import { packTensor } from '@base/utilities/packed';
import { engine, Tensor } from '@tensorflow/tfjs-core';

import './grads/pack16';

export function pack16(x: Tensor, scaling = 1): Tensor {
    const t = engine().runKernel('Pack16', { x }, { scaling }) as Tensor;
    packTensor(t);
    return t;
}
