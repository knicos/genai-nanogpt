import { GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';
import { dSilu } from '../silu';

export const siluGradConfig: GradConfig = {
    kernelName: 'Silu',
    inputsToSave: ['x'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        const [x] = saved;

        return {
            x: () => dSilu(dy as Tensor, x),
        };
    },
};

registerGradient(siluGradConfig);
