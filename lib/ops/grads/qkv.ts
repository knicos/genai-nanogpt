import { registerGradient, GradConfig, Tensor } from '@tensorflow/tfjs-core';

const qkvGradConfig: GradConfig = {
    kernelName: 'QKV',
    inputsToSave: ['x', 'kernel'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        // dy: [dq, dk, dv]
        // x: input tensor
        // kernel: fused kernel weights
        const [dq, dk, dv] = dy as Tensor[];
        const [x, kernel] = saved as Tensor[];

        // Get shapes
        const [B, T, C] = x.shape;

        // Reshape dy to [B*T, C] for each Q, K, V
        const dq2d = dq.transpose([0, 2, 1, 3]).reshape([B * T, C]);
        const dk2d = dk.transpose([0, 2, 1, 3]).reshape([B * T, C]);
        const dv2d = dv.transpose([0, 2, 1, 3]).reshape([B * T, C]);

        // Gradient w.r.t x: sum of matMul with each kernel slice
        const kernelQ = kernel.slice([0, 0], [C, C]);
        const kernelK = kernel.slice([0, C], [C, C]);
        const kernelV = kernel.slice([0, 2 * C], [C, C]);

        return {
            x: () => {
                const dxQ = dq2d.matMul(kernelQ, false, true);
                const dxK = dk2d.matMul(kernelK, false, true);
                const dxV = dv2d.matMul(kernelV, false, true);
                const dx = dxQ.add(dxK).add(dxV).reshape([B, T, C]);
                return dx;
            },
            kernel: () => {
                // Gradient w.r.t kernel: x^T * dy for each Q, K, V
                const x2d = x.reshape([B * T, C]);
                const dKernelQ = x2d.matMul(dq2d, true, false);
                const dKernelK = x2d.matMul(dk2d, true, false);
                const dKernelV = x2d.matMul(dv2d, true, false);
                const dKernel = dKernelQ.concat(dKernelK, 1).concat(dKernelV, 1); // [C, 3*C]
                return dKernel;
            },
        };
    },
};

registerGradient(qkvGradConfig);
