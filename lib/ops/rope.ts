import RoPECache from '@base/layers/RoPECache';
import { NamedAttrMap, Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/rope';
import './webgl/rope';
import './grads/rope';

export function rope(x: Tensor, cache: RoPECache, pastLength: number, negSin = false): Tensor {
    cache.ensureRopeCache(x.shape[1]! + pastLength); // x.shape[1] = Tcur
    const r = engine().runKernel('Rope', { x }, {
        pastLen: pastLength,
        negSin,
        ropeCache: cache,
    } as unknown as NamedAttrMap) as Tensor;
    return r;
}
