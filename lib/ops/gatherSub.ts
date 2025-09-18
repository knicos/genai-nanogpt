import { Tensor, engine } from '@tensorflow/tfjs-core';
import './cpu/gatherSub';
import './webgl/gatherSub';

export function gatherSub(values: Tensor, labels: Tensor, logits: Tensor): Tensor {
    return engine().runKernel('EfficientGatherSub', { logits, labels, values }, {});
}
