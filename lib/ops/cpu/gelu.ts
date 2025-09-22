import {
    KernelConfig,
    KernelFunc,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
    tidy,
} from '@tensorflow/tfjs-core';

const K = 0.7978845608028654; // sqrt(2/pi)
const A = 0.044715;

// Approximate GELU (GPT-2 uses this)
function gelu(args: { inputs: { x: TensorInfo }; backend: unknown }): TensorInfo {
    const { inputs } = args;
    const { x: ti } = inputs;
    const x = ti as Tensor;

    // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    return tidy(() => {
        const x3 = x.mul(x).mul(x);
        const inner = x.add(x3.mul(A)).mul(K).tanh().add(1).mul(0.5);
        return x.mul(inner);
    });
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'Gelu',
    backendName: 'cpu',
    kernelFunc: gelu as unknown as KernelFunc,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'Gelu',
    backendName: 'tensorflow',
    kernelFunc: gelu as unknown as KernelFunc,
};

registerKernel(tensorflowKernelConfig);

// Backward

function geluGradKernelFunc(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { dy, x } = args.inputs as { dy: Tensor; x: Tensor };

    return tidy(() => {
        // u = k * (x + 0.044715 * x^3)
        const x2 = x.square();
        const x3 = x2.mul(x);
        const u = x.add(x3.mul(A)).mul(K);
        const tanhU = u.tanh();
        const sech2 = tanhU.square().neg().add(1); // 1 - tanh(u)^2

        // 1 + 3 * a * x^2
        const inner = x2.mul(3 * A).add(1);

        // dgelu/dx
        const left = tanhU.add(1).mul(0.5);
        const right = x.mul(sech2).mul(K).mul(inner).mul(0.5);
        const grad = left.add(right);

        return (dy as Tensor).mul(grad);
    });
}

const geluGradKernelConfig: KernelConfig = {
    kernelName: 'GeluGrad',
    backendName: 'cpu',
    kernelFunc: geluGradKernelFunc,
};

registerKernel(geluGradKernelConfig);

const tensorflowGradKernelConfig: KernelConfig = {
    kernelName: 'GeluGrad',
    backendName: 'tensorflow',
    kernelFunc: geluGradKernelFunc,
};

registerKernel(tensorflowGradKernelConfig);
