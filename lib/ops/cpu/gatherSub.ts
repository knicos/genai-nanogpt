import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    range,
    stack,
    sub,
    gatherND,
    Tensor,
} from '@tensorflow/tfjs-core';

// CPU fallback implementation
function gatherSubCPU(args: { inputs: NamedTensorInfoMap }): TensorInfo {
    const { values, labels, logits } = args.inputs as { values: Tensor; labels: Tensor; logits: Tensor };
    const batchSize = labels.shape[0];
    const batchIndices = range(0, batchSize, 1, 'int32');
    const indices = stack([batchIndices, labels], 1);
    const correctLogits = gatherND(logits, indices);

    // Cross-entropy loss: -correctLogits + logSumExp
    return sub(values, correctLogits); //.as1D();
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'EfficientGatherSub',
    backendName: 'cpu',
    kernelFunc: gatherSubCPU,
};

registerKernel(cpuKernelConfig);
