import { KernelConfig, KernelFunc, registerKernel, Tensor, TensorInfo, tidy } from '@tensorflow/tfjs-core';

function matMulMul(args: {
    inputs: { x: TensorInfo; kernel: TensorInfo; y: TensorInfo };
    attrs: { transposeA: boolean; transposeB: boolean };
    backend: unknown;
}): TensorInfo {
    const { inputs, attrs } = args;
    const { transposeA, transposeB } = attrs;
    const { x: ti, kernel: ki, y: yi } = inputs;
    const x = ti as Tensor;
    const kernel = ki as Tensor;
    const y = yi as Tensor;

    return tidy(() => {
        const m = x.matMul(kernel, transposeA, transposeB);
        return m.mul(y);
    });
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'MatMulMul',
    backendName: 'cpu',
    kernelFunc: matMulMul as unknown as KernelFunc,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'MatMulMul',
    backendName: 'tensorflow',
    kernelFunc: matMulMul as unknown as KernelFunc,
};

registerKernel(tensorflowKernelConfig);
