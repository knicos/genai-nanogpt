import {
    KernelConfig,
    KernelFunc,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
    tidy,
} from '@tensorflow/tfjs-core';

// Approximate GELU (GPT-2 uses this)
function normRMS(args: { inputs: { x: TensorInfo; gamma: TensorInfo }; backend: unknown }): TensorInfo {
    const { inputs } = args;
    const { x: ti, gamma: gi } = inputs;
    const x = ti as Tensor;
    const gamma = gi as Tensor;

    // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    return tidy(() => {
        const meanSquare = x.square().mean(-1, true);
        const invRms = meanSquare.add(1e-8).rsqrt();
        const normalized = x.mul(invRms);
        const result = normalized.mul(gamma);
        return result;
    });
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'RMSNorm',
    backendName: 'cpu',
    kernelFunc: normRMS as unknown as KernelFunc,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'RMSNorm',
    backendName: 'tensorflow',
    kernelFunc: normRMS as unknown as KernelFunc,
};

registerKernel(tensorflowKernelConfig);

// Backward

function normRMSGradKernelFunc(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo[] {
    const { dy, x, gamma } = args.inputs as { dy: Tensor; x: Tensor; gamma: Tensor };

    return tidy(() => {
        const N = x.shape[x.shape.length - 1];
        const meanSquare = x.square().mean(-1, true);
        const invRms = meanSquare.add(1e-8).rsqrt();
        const normed = x.mul(invRms); // shape [batch, T, C]

        // Gradient w.r.t. gamma: sum_{b,t} dy_{b,t,c} * normed_{b,t,c}
        const gammaGrad = dy.mul(normed).sum([0, 1]);

        // dy * gamma
        const dyGamma = dy.mul(gamma);

        // sum_j (dy_j * x_j)
        const dyXMean = dyGamma.mul(x).sum(-1, true).div(N);

        // dx = gamma * invRms * [dy - x * dyXMean / (meanSquare + eps)]
        const dx = dyGamma.mul(invRms).sub(x.mul(dyXMean).mul(invRms).div(meanSquare.add(1e-8)));

        return [dx, gammaGrad];
    });
}

const normRMSGradKernelConfig: KernelConfig = {
    kernelName: 'RMSNormGrad',
    backendName: 'cpu',
    kernelFunc: normRMSGradKernelFunc,
};

registerKernel(normRMSGradKernelConfig);

const tensorflowGradKernelConfig: KernelConfig = {
    kernelName: 'RMSNormGrad',
    backendName: 'tensorflow',
    kernelFunc: normRMSGradKernelFunc,
};

registerKernel(tensorflowGradKernelConfig);
