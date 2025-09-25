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

        return {
            x: () => {
                const kernelQ = kernel.slice([0, 0], [C, C]);
                const dxQ = dq2d.matMul(kernelQ, false, true);
                kernelQ.dispose();

                const kernelK = kernel.slice([0, C], [C, C]);
                const dxK = dk2d.matMul(kernelK, false, true);
                kernelK.dispose();
                const dx1 = dxQ.add(dxK);
                dxQ.dispose();
                dxK.dispose();

                const kernelV = kernel.slice([0, 2 * C], [C, C]);
                const dxV = dv2d.matMul(kernelV, false, true);
                kernelV.dispose();

                const dx = dx1.add(dxV).reshape([B, T, C]);
                dx1.dispose();
                dxV.dispose();
                return dx;
            },
            kernel: () => {
                // Gradient w.r.t kernel: x^T * dy for each Q, K, V
                const x2d = x.reshape([B * T, C]);
                const dKernelQ = x2d.matMul(dq2d, true, false);
                const dKernelK = x2d.matMul(dk2d, true, false);
                const dKernel1 = dKernelQ.concat(dKernelK, 1); // [C, 2*C]
                dKernelQ.dispose();
                dKernelK.dispose();
                const dKernelV = x2d.matMul(dv2d, true, false);
                const dKernel = dKernel1.concat(dKernelV, 1); // [C, 3*C]
                dKernel1.dispose();
                dKernelV.dispose();
                x2d.dispose();
                return dKernel;
            },
        };
    },
};

registerGradient(qkvGradConfig);
