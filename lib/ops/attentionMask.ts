import { Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/attentionMask';
import './webgl/attentionMask';
import './grads/attentionMask';

export function attentionMask(q: Tensor, k: Tensor, mask: Tensor, divisor: number): Tensor {
    return engine().runKernel('AttentionMask', { q, k, mask }, { divisor });
}
