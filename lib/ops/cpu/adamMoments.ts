import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
    stack,
} from '@tensorflow/tfjs-core';

// CPU fallback implementation
function adamMomentsCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo {
    const { moments, gradient } = args.inputs as { moments: Tensor; gradient: Tensor };
    const { beta1, beta2 } = args.attrs as { beta1: number; beta2: number };

    const rank = moments.shape.length;

    // Slicing along the last axis for m1 and m2
    const beginM1 = new Array(rank).fill(0);
    const sizeM1 = moments.shape.slice();
    sizeM1[rank - 1] = 1;

    const beginM2 = beginM1.slice();
    beginM2[rank - 1] = 1;
    const sizeM2 = sizeM1.slice();

    const m1 = moments.slice(beginM1, sizeM1).squeeze([rank - 1]);
    const m2 = moments.slice(beginM2, sizeM2).squeeze([rank - 1]);

    const newM1 = m1.mul(beta1).add(gradient.mul(1 - beta1));
    const newM2 = m2.mul(beta2).add(gradient.square().mul(1 - beta2));

    const newMoments = stack([newM1, newM2], -1);

    return newMoments;
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'AdamMoments',
    backendName: 'cpu',
    kernelFunc: adamMomentsCPU,
};

registerKernel(cpuKernelConfig);
