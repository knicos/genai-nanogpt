import { GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';
import { unpack16 } from '../unpack16';

export const packGradConfig: GradConfig = {
    kernelName: 'Pack16',
    inputsToSave: [],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[]) => {
        return {
            x: () => {
                return unpack16(dy as Tensor);
            },
        };
    },
};

registerGradient(packGradConfig);
