import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
    div,
    add,
    mul,
    sqrt,
} from '@tensorflow/tfjs-core';

// CPU fallback implementation
function adamAdjustCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo {
    const { moments, value } = args.inputs as { moments: Tensor; value: Tensor };
    const { beta1, beta2, epsilon, learningRate } = args.attrs as {
        beta1: number;
        beta2: number;
        epsilon: number;
        learningRate: number;
    };

    const rank = moments.shape.length;
    const beginM1 = new Array(rank).fill(0);
    const sizeM1 = moments.shape.slice();
    sizeM1[rank - 1] = 1;

    const beginM2 = beginM1.slice();
    beginM2[rank - 1] = 1;
    const sizeM2 = sizeM1.slice();

    const m1 = moments.slice(beginM1, sizeM1).squeeze([rank - 1]);
    const m2 = moments.slice(beginM2, sizeM2).squeeze([rank - 1]);

    const biasCorrectedFirstMoment = div(m1, beta1);
    const biasCorrectedSecondMoment = div(m2, beta2);

    const newValue = add(
        mul(div(biasCorrectedFirstMoment, add(sqrt(biasCorrectedSecondMoment), epsilon ?? 1e-8)), -learningRate),
        value
    );

    return newValue;
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'AdamAdjust',
    backendName: 'cpu',
    kernelFunc: adamAdjustCPU,
};

registerKernel(cpuKernelConfig);
