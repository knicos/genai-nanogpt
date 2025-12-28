import { isPackedTensor } from '@base/utilities/packed';
import { concat, engine, NamedTensorMap, Rank, Tensor } from '@tensorflow/tfjs-core';

export function concat16<R extends Rank = Rank>(x: Tensor<R>[], axis?: number): Tensor<R> {
    const packed = isPackedTensor(x[0]);

    if (!packed) {
        return concat(x, axis);
    }

    return engine().runKernel('Concat16', x as unknown as NamedTensorMap, { axis: axis ?? -1 }) as Tensor<R>;
}
