import type { Rank } from '@tensorflow/tfjs-core/dist/types';
import { Tensor, Variable } from '@tensorflow/tfjs-core/dist/tensor';
import { TensorInfo } from '@tensorflow/tfjs-core/dist/tensor_info';

export interface PackedTensorInfo extends TensorInfo {
    packed?: boolean;
}

export class PackableTensor<R extends Rank = Rank> extends Tensor<R> implements PackedTensorInfo {
    packed: boolean = false;
}

export class PackableVariable<R extends Rank = Rank> extends Variable<R> implements PackedTensorInfo {
    packed: boolean = false;
}
