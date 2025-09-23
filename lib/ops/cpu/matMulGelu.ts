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
function matMulGelu(args: { inputs: { x: TensorInfo; kernel: TensorInfo }; backend: unknown }): TensorInfo {
    const { inputs } = args;
    const { x: ti, kernel: ki } = inputs;
    const x = ti as Tensor;
    const kernel = ki as Tensor;

    // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    return tidy(() => {
        const m = x.matMul(kernel);
        const m3 = m.mul(m).mul(m);
        const inner = m.add(m3.mul(A)).mul(K).tanh().add(1).mul(0.5);
        return m.mul(inner);
    });
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'MatMulGelu',
    backendName: 'cpu',
    kernelFunc: matMulGelu as unknown as KernelFunc,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'MatMulGelu',
    backendName: 'tensorflow',
    kernelFunc: matMulGelu as unknown as KernelFunc,
};

registerKernel(tensorflowKernelConfig);

// Backward

function matMulGeluGradKernelFunc(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo[] {
    const { dy, x, kernel } = args.inputs as { dy: Tensor; x: Tensor; kernel: Tensor };

    return tidy(() => {
        // From forward pass
        const m = x.matMul(kernel);

        // u = k * (x + 0.044715 * x^3)
        const m2 = m.square();
        const m3 = m2.mul(m);
        const u = m.add(m3.mul(A)).mul(K);
        const tanhU = u.tanh();
        const sech2 = tanhU.square().neg().add(1); // 1 - tanh(u)^2

        // 1 + 3 * a * x^2
        const inner = m2.mul(3 * A).add(1);

        // dgelu/dx
        const left = tanhU.add(1).mul(0.5);
        const right = m.mul(sech2).mul(K).mul(inner).mul(0.5);
        const grad = left.add(right);

        const dL_dm = (dy as Tensor).mul(grad); // dL/dm = dL/dy * dy/dm

        // Gradients w.r.t. x and kernel
        const dx = dL_dm.matMul(kernel.transpose());
        const dkernel = x.transpose().matMul(dL_dm);
        return [dx, dkernel];
    });
}

const matMulGeluGradKernelConfig: KernelConfig = {
    kernelName: 'MatMulGeluGrad',
    backendName: 'cpu',
    kernelFunc: matMulGeluGradKernelFunc,
};

registerKernel(matMulGeluGradKernelConfig);

const tensorflowGradKernelConfig: KernelConfig = {
    kernelName: 'MatMulGeluGrad',
    backendName: 'tensorflow',
    kernelFunc: matMulGeluGradKernelFunc,
};

registerKernel(tensorflowGradKernelConfig);
