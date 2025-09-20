import {
    KernelConfig,
    KernelFunc,
    registerKernel,
    softmax,
    SoftmaxAttrs,
    SoftmaxInputs,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';

interface FusedSoftmaxAttrs extends SoftmaxAttrs {
    dropoutRate?: number;
}

export function softmaxCPU(args: { inputs: SoftmaxInputs; attrs: FusedSoftmaxAttrs }): TensorInfo {
    const { inputs, attrs } = args;
    const { logits } = inputs;
    const { dim, dropoutRate } = attrs;

    if (!logits) {
        throw new Error('Error in softmax: input logits is null');
    }

    if (dropoutRate !== undefined && dropoutRate > 0) {
        /*const droppedRes = dropout(res, dropoutRate);
        maxLogit.dispose();
        maxLogitsReshaped.dispose();
        a.dispose();
        b.dispose();
        sumExp.dispose();
        sumExpReshaped.dispose();
        res.dispose();
        return droppedRes;*/

        console.warn('Dropout in fusedSoftmax not implemented for CPU backend, skipping dropout.');
    }

    return softmax(logits as Tensor, dim);
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'FusedSoftmax',
    backendName: 'cpu',
    kernelFunc: softmaxCPU as unknown as KernelFunc,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'FusedSoftmax',
    backendName: 'tensorflow',
    kernelFunc: softmaxCPU as unknown as KernelFunc,
};

registerKernel(tensorflowKernelConfig);
