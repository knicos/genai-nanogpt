import { engine, Tensor } from '@tensorflow/tfjs-core';

import './cpu/qkv';
import './webgl/qkv';
import './grads/qkv';

export function qkv(x: Tensor, kernel: Tensor, heads: number, packed = false): Tensor[] {
    const r = engine().runKernel('QKV', { x, kernel }, { heads, packed }) as Tensor[];
    return r;
}
