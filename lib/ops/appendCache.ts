import { Tensor, engine } from '@tensorflow/tfjs-core';

import './cpu/appendCache';
import './webgl/appendCache';

export function appendCache(cache: Tensor, item: Tensor, maxSize: number): Tensor {
    return engine().runKernel('AppendCache', { cache, item }, { maxSize });
}

/*const appendCacheGradConfig: GradConfig = {
    kernelName: 'AppendCache',
    inputsToSave: ['cache'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        if (Array.isArray(dy)) {
            throw new Error('Expected single tensor input.');
        }
        const cache = saved[0];
        const T = cache.shape[2]!; // original sequence length

        // dy: [B, nh, outT, hs], outT = min(T+1, maxSize)
        // cache: [B, nh, T, hs]
        // item: [B, nh, 1, hs]

        // Gradient for cache: first T elements along axis 2
        const dCache = dy.slice([0, 0, 0, 0], [dy.shape[0], dy.shape[1]!, T, dy.shape[3]!]);
        // Gradient for item: last element along axis 2
        const dItem = dy.slice([0, 0, dy.shape[2]! - 1, 0], [dy.shape[0], dy.shape[1]!, 1, dy.shape[3]!]);

        return { cache: () => dCache, item: () => dItem };
    },
};

registerGradient(appendCacheGradConfig);*/
