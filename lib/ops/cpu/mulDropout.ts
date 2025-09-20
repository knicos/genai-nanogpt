import { KernelConfig, KernelFunc, mul, registerKernel, Tensor, TensorInfo } from '@tensorflow/tfjs-core';

interface MulDropoutAttrs {
    dropoutRate?: number;
    seed?: number;
}

function mulDrop(args: {
    inputs: { a: TensorInfo; b: TensorInfo };
    backend: unknown;
    attrs: MulDropoutAttrs;
}): TensorInfo {
    const { inputs } = args;
    const { a, b } = inputs;

    console.warn('Using fallback mulDrop implementation without dropout.');

    return mul(a as Tensor, b as Tensor);
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'MulDropout',
    backendName: 'cpu',
    kernelFunc: mulDrop as unknown as KernelFunc,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'MulDropout',
    backendName: 'tensorflow',
    kernelFunc: mulDrop as unknown as KernelFunc,
};

registerKernel(tensorflowKernelConfig);
