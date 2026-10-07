import RoPECache from '@base/layers/RoPECache';
import { NamedAttrMap, Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/rope';
import './webgl/rope';
import './grads/rope';

export function rope(x: Tensor, cache: RoPECache, pastLength: number, negSin = false, interleaved = true): Tensor {
    cache.ensureRopeCache(x.shape[2]! + pastLength); // x.shape[1] = Tcur
    const r = engine().runKernel('Rope', { x }, {
        pastLen: pastLength,
        negSin,
        ropeCache: cache,
        interleaved,
    } as unknown as NamedAttrMap) as Tensor;
    return r;
}
