import { GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';
import { dGelu } from '../gelu';

export const geluGradConfig: GradConfig = {
    kernelName: 'Gelu',
    inputsToSave: ['x'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        const [x] = saved;

        return {
            x: () => dGelu(dy as Tensor, x),
        };
    },
};

registerGradient(geluGradConfig);
