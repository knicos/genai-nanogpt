import { isPackedTensor } from '@base/utilities/packed';
import { engine, Rank, slice, Tensor } from '@tensorflow/tfjs-core';

export function slice16<R extends Rank = Rank>(
    x: Tensor<R>,
    begin: number | number[],
    size: number | number[]
): Tensor<R> {
    const packed = isPackedTensor(x);

    if (!packed) {
        return slice(x, begin, size);
    }

    return engine().runKernel('Slice16', { x }, { begin, size }) as Tensor<R>;
}
