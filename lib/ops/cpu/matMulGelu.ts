import {
    KernelConfig,
    KernelFunc,
    matMul,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
    tidy,
} from '@tensorflow/tfjs-core';
import { dGelu, gelu } from '../gelu';

function matMulGelu(args: { inputs: { x: TensorInfo; kernel: TensorInfo }; backend: unknown }): TensorInfo {
    const { inputs } = args;
    const { x: ti, kernel: ki } = inputs;
    const x = ti as Tensor;
    const kernel = ki as Tensor;

    // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
    return tidy(() => {
        const m = matMul(x, kernel);
        /*const m3 = m.mul(m).mul(m);
        const inner = m.add(m3.mul(A)).mul(K).tanh().add(1).mul(0.5);
        return m.mul(inner);*/
        return gelu(m);
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

const webgpuKernelConfig: KernelConfig = {
    kernelName: 'MatMulGelu',
    backendName: 'webgpu',
    kernelFunc: matMulGelu as unknown as KernelFunc,
};

registerKernel(webgpuKernelConfig);

// Backward

function matMulGeluGradKernelFunc(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo[] {
    const { dy, x, kernel } = args.inputs as { dy: Tensor; x: Tensor; kernel: Tensor };

    return tidy(() => {
        // From forward pass
        const m = matMul(x, kernel);
        const dL_dm = dGelu(dy, m);

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

const webgpuGradKernelConfig: KernelConfig = {
    kernelName: 'MatMulGeluGrad',
    backendName: 'webgpu',
    kernelFunc: matMulGeluGradKernelFunc,
};

registerKernel(webgpuGradKernelConfig);
