import { Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/attentionMask';
import './webgl/attentionMask';
import './grads/attentionMask';

export function attentionMask(q: Tensor, k: Tensor, divisor: number, pastLen?: number): Tensor {
    const t = engine().runKernel('AttentionMask', { q, k }, { divisor, pastLen: pastLen || 0 }) as Tensor;
    //packTensor(t);
    return t;
}
