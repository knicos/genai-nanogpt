import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    range,
    stack,
    ones,
    scatterND,
    sub,
    mul,
    Tensor,
} from '@tensorflow/tfjs-core';

// CPU fallback implementation
function efficientScatterSubCPU(args: { inputs: NamedTensorInfoMap }): TensorInfo {
    const { logits, labels, dy } = args.inputs as { logits: Tensor; labels: Tensor; dy: Tensor };

    const batchSize = labels.shape[0];
    const depth = logits.shape[1]!;
    const batchIndices = range(0, batchSize, 1, 'int32');
    const indices = stack([batchIndices, labels], 1);
    const updates = ones([batchSize]);
    const subtractTensor = scatterND(indices, updates, [batchSize, depth]);
    const gradLogits = sub(logits, subtractTensor);
    const dyReshaped = dy.reshape([batchSize, 1]);
    const gradLogitsScaled = mul(gradLogits, dyReshaped);
    return gradLogitsScaled;
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'EfficientScatterSub',
    backendName: 'cpu',
    kernelFunc: efficientScatterSubCPU,
};

registerKernel(cpuKernelConfig);
