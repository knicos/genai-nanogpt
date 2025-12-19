import { engine, Tensor } from '@tensorflow/tfjs-core';

import './cpu/qkv';
import './webgl/qkv';
import './grads/qkv';
import { packTensor } from '@base/utilities/packed';

export function qkv(x: Tensor, kernel: Tensor, heads: number, packed = false): Tensor[] {
    const r = engine().runKernel('QKV', { x, kernel }, { heads, packed }) as Tensor[];
    if (packed) {
        r.forEach((t) => {
            packTensor(t);
        });
    }
    return r;
}
