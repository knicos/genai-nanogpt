import { engine, GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';

function dNormRMS(dy: Tensor, x: Tensor, gamma?: Tensor): Tensor[] {
    return engine().runKernel('RMSNormGrad', gamma ? { dy, x, gamma } : { dy, x });
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

export const normRMSGradConfigNoGamma: GradConfig = {
    kernelName: 'RMSNormNoGamma',
    inputsToSave: ['x'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        const [x] = saved;

        const [dx] = dNormRMS(dy as Tensor, x);

        return {
            x: () => dx,
        };
    },
};

registerGradient(normRMSGradConfigNoGamma);
