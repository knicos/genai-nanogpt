import { engine, GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';

function dNormRMS(dy: Tensor, x: Tensor, gamma: Tensor): Tensor[] {
    return engine().runKernel('RMSNormGrad', { dy, x, gamma });
}

export const normRMSGradConfig: GradConfig = {
    kernelName: 'RMSNorm',
    inputsToSave: ['x', 'gamma'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        const [x, gamma] = saved;

        const [dx, dgamma] = dNormRMS(dy as Tensor, x, gamma);

        return {
            x: () => dx,
            gamma: () => dgamma,
        };
    },
};

registerGradient(normRMSGradConfig);
