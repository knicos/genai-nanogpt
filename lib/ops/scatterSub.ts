import { Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/scatterSub';
import './webgl/scatterSub';

export function scatterSub(probabilities: Tensor, labels: Tensor, scale: Tensor): Tensor {
    return engine().runKernel('EfficientScatterSub', { logits: probabilities, labels, dy: scale }, {});
}
