import { isPackedTensor, packTensor } from '@base/utilities/packed';
import { Rank, slice, Tensor } from '@tensorflow/tfjs-core';

export function slice16<R extends Rank = Rank>(
    x: Tensor<R>,
    begin: number | number[],
    size: number | number[]
): Tensor<R> {
    const r = slice(x, begin, size);
    if (isPackedTensor(x)) {
        packTensor(r);
    }
    return r;
}
