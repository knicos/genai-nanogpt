import { GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';
import { pack16 } from '../pack16';

export const unpackGradConfig: GradConfig = {
    kernelName: 'Unpack16',
    inputsToSave: [],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[]) => {
        return {
            x: () => {
                return pack16(dy as Tensor);
            },
        };
    },
};

registerGradient(unpackGradConfig);
