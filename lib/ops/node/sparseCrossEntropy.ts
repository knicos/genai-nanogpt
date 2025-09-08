import { registerKernel, KernelConfig, TensorInfo, NamedTensorInfoMap } from '@tensorflow/tfjs-core';
import { NodeJSKernelBackend } from '@tensorflow/tfjs-node-gpu/dist/nodejs_kernel_backend';

function nativeSparseSoftmaxCrossEntropy(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo[] {
    const { logits, labels } = args.inputs as { logits: TensorInfo; labels: TensorInfo };
    const backend = args.backend as NodeJSKernelBackend;
    // This calls the native TensorFlow op
    return backend.executeMultipleOutputs('SparseSoftmaxCrossEntropyWithLogits', [], [logits, labels], 2);
}

const kernelConfig: KernelConfig = {
    kernelName: 'NativeSparseSoftmaxCrossEntropy',
    backendName: 'tensorflow',
    kernelFunc: nativeSparseSoftmaxCrossEntropy,
};

registerKernel(kernelConfig);
