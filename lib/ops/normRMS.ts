import { engine, Tensor } from '@tensorflow/tfjs-core';

import './cpu/normRMS';
import './webgl/normRMS';
import './grads/normRMS';

export function normRMS(x: Tensor, gamma?: Tensor): Tensor {
    // If x is rank 4, reduce to rank 3 by merging the first two dimensions (batch and sequence length)
    if (x.rank === 4) {
        const [batch, seqLen, ...rest] = x.shape;
        const newShape = [batch * seqLen, ...rest];
        const reshapedX = x.reshape(newShape);
        const result = engine().runKernel(
            gamma ? 'RMSNorm' : 'RMSNormNoGamma',
            gamma ? { x: reshapedX, gamma } : { x: reshapedX }
        ) as Tensor;
        reshapedX.dispose();
        const out = result.reshape(x.shape);
        result.dispose();
        return out;
    }
    return engine().runKernel(gamma ? 'RMSNorm' : 'RMSNormNoGamma', gamma ? { x, gamma } : { x });
}
