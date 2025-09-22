import { Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/attentionMask';
import './webgl/attentionMask';
import './grads/attentionMask';

export function attentionMask(q: Tensor, k: Tensor, divisor: number, mask?: Tensor, pastLen?: number): Tensor {
    if (mask) {
        return engine().runKernel('AttentionMask', { q, k, mask }, { divisor, pastLen: pastLen || 0 });
    }
    return engine().runKernel('AttentionMask', { q, k }, { divisor, pastLen: pastLen || 0 });
}
