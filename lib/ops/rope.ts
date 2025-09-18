import RoPECache from '@base/layers/RoPECache';
import { Tensor, engine } from '@tensorflow/tfjs';

import './cpu/rope';
import './webgl/rope';
import './grads/rope';

export function rope(x: Tensor, cache: RoPECache, pastLength: number): Tensor {
    cache.ensureRopeCache(x.shape[1]! + pastLength); // x.shape[1] = Tcur
    return engine().runKernel('Rope', { x, sin: cache.getSin()!, cos: cache.getCos()! }, { pastLen: pastLength });
}
