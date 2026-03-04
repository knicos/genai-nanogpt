import { KernelConfig, NamedTensorInfoMap, registerKernel, Tensor, TensorInfo } from '@tensorflow/tfjs-core';

function dropoutGradKernelFunc(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    return x;
}

const dropout16GradKernelConfig: KernelConfig = {
    kernelName: 'Dropout16',
    backendName: 'webgl',
    kernelFunc: dropoutGradKernelFunc,
};

registerKernel(dropout16GradKernelConfig);
