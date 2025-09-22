import { Tensor, concat, engine, zeros } from '@tensorflow/tfjs-core';

import './cpu/appendCache';
import './webgl/appendCache';

export function appendCache(item: Tensor, maxSize: number, pastLen: number, cache?: Tensor): Tensor {
    if (!cache) {
        const Tcur = item.shape[2]!;
        return concat([item, zeros([item.shape[0]!, item.shape[1]!, maxSize - Tcur, item.shape[3]!])], 2);
    }
    return engine().runKernel('AppendCache', { cache, item }, { maxSize, pastLen });
}
