import {
    engine,
    GradConfig,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerGradient,
    registerKernel,
    reshape,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { isPackedTensor, packTensor } from '@base/utilities/packed';

const reshapeGradConfig: GradConfig = {
    kernelName: 'Reshape16',
    inputsToSave: ['x'],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        const [x] = saved;
        if (Array.isArray(dy)) {
            throw new Error('Reshape16 gradient does not support multiple outputs.');
        }
        return { x: () => reshape16(dy, x.shape) };
    },
};

registerGradient(reshapeGradConfig);

function reshape16_(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { inputs, attrs } = args;
    const { x } = inputs as { x: Tensor };
    const { shape } = attrs as unknown as { shape: number[] };

    const packed = isPackedTensor(x);

    if (!packed) {
        const result = reshape(x as Tensor, shape);
        return result;
    }

    const result = packTensor(reshape(x as Tensor, shape));
    return result;
}

const webgpuConfig: KernelConfig = {
    kernelName: 'Reshape16',
    backendName: 'webgpu',
    kernelFunc: reshape16_,
};

registerKernel(webgpuConfig);

const webglConfig: KernelConfig = {
    kernelName: 'Reshape16',
    backendName: 'webgl',
    kernelFunc: reshape16_,
};

registerKernel(webglConfig);

const cpuConfig: KernelConfig = {
    kernelName: 'Reshape16',
    backendName: 'cpu',
    kernelFunc: reshape16_,
};

registerKernel(cpuConfig);

export function reshape16(x: Tensor, shape: number[]): Tensor {
    const r = engine().runKernel('Reshape16', { x }, { shape }) as Tensor;
    //packTensor(r);
    return r;
}
