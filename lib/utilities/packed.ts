import { engine, type Tensor } from '@tensorflow/tfjs-core';

export function packingSupported(): boolean {
    return engine().backendName === 'webgpu';
}

// Type guard
export function isPackableTensor(tensor: Tensor): boolean {
    return tensor.dtype === 'packedF16';
}

export function isPackedTensor(tensor: Tensor): boolean {
    return isPackableTensor(tensor);
}
