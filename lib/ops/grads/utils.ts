import { TensorInfo } from '@tensorflow/tfjs-core';

export function forceFloat<T extends TensorInfo>(x: T): T {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (x.dtype as any) = 'float32';
    return x;
}

export function forceInt<T extends TensorInfo>(x: T): T {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (x.dtype as any) = 'int32';
    return x;
}
