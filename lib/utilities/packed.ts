import { ENGINE } from '@base/patches/engine';
import type { PackableTensor } from '@base/patches/PackedTensor';
import { Tensor } from '@tensorflow/tfjs-core';

export function packingSupported(): boolean {
    return ENGINE.backendName === 'webgpu';
}

// Type guard
export function isPackableTensor(tensor: Tensor): tensor is PackableTensor {
    return (tensor as PackableTensor).packed !== undefined;
}

export function isPackedTensor(tensor: Tensor): boolean {
    return isPackableTensor(tensor) && (tensor as PackableTensor).packed;
}

export function packTensor(tensor: Tensor): Tensor {
    if (isPackableTensor(tensor)) {
        if (tensor.dtype !== 'int32') {
            throw new Error('packTensor: only int32 tensors can be packed.');
        }
        (tensor as PackableTensor).packed = true;
        return tensor;
    } else {
        console.error('Tensor:', tensor);
        throw new Error('Tensor is not packable');
    }
}

export function unpackTensor(tensor: Tensor): Tensor {
    if (isPackableTensor(tensor)) {
        if (tensor.dtype !== 'float32') {
            throw new Error('unpackTensor: only float32 tensors can be unpacked.');
        }
        (tensor as PackableTensor).packed = false;
    }
    return tensor;
}
