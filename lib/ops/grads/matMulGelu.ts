import { engine, GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';

function dMatMulGelu(dy: Tensor, x: Tensor, kernel: Tensor): Tensor[] {
    return engine().runKernel('MatMulGeluGrad', { dy, x, kernel });
}

const matMulGeluGradConfig: GradConfig = {
    kernelName: 'MatMulGelu',
    inputsToSave: ['x', 'kernel'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        const [x, kernel] = saved;

        const [dx, dKernel] = dMatMulGelu(dy as Tensor, x, kernel);

        return {
            x: () => dx,
            kernel: () => dKernel,
        };
    },
};

registerGradient(matMulGeluGradConfig);
