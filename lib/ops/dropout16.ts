import { isPackedTensor } from '@base/utilities/packed';
import { engine, Rank, Tensor } from '@tensorflow/tfjs-core';

import './grads/dropout16';
import './webgl/dropout16';

export function dropout16<R extends Rank = Rank>(x: Tensor<R>, dropout: number, seed?: number): Tensor<R> {
    if (!isPackedTensor(x)) {
        return x;
    }
    return engine().runKernel(
        'Dropout16',
        { x },
        {
            dropout,
            seed: seed ?? Math.random(),
        }
    ) as Tensor<R>;
}
