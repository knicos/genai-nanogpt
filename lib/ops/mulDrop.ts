import { engine, Tensor } from '@tensorflow/tfjs-core';

import './cpu/mulDropout';
import './webgl/mulDropout';

export function mulDrop(a: Tensor, b: Tensor, dropoutRate: number, seed: number): Tensor {
    return engine().runKernel('MulDropout', { a, b }, { dropoutRate, seed });
}
