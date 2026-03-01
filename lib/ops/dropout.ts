import { add, div, floor, randomUniform, Tensor, TensorLike, util } from '@tensorflow/tfjs-core';
import { convertToTensor } from '@tensorflow/tfjs-core/dist/tensor_util_env';
import { getNoiseShape } from '@tensorflow/tfjs-core/dist/ops/dropout_util';
import { mul16 } from './mul16';

export function dropout(x: Tensor | TensorLike, rate: number, noiseShape?: number[], seed?: number | string): Tensor {
    const $x = convertToTensor(x, 'x', 'dropout');

    util.assert(rate >= 0 && rate < 1, () => `rate must be a float in the range [0, 1), but got ${rate}.`);

    if (rate === 0) {
        return x instanceof Tensor ? $x.clone() : $x;
    }

    const $noiseShape = getNoiseShape($x, noiseShape);
    const keepProb = 1 - rate;
    const multiplier = div(floor(add(randomUniform($noiseShape, 0, 1, 'float32', seed), keepProb)), keepProb);

    return mul16($x, multiplier);
}
