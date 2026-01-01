import { GradConfig, registerGradient, slice, Tensor } from '@tensorflow/tfjs-core';
import { unpack16 } from '../unpack16';

export const packGradConfig: GradConfig = {
    kernelName: 'Pack16',
    inputsToSave: [],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], _, attrs: { originalShape?: number[]; padding?: number }) => {
        return {
            x: () => {
                const unpacked = unpack16(dy as Tensor);
                if (attrs.originalShape && attrs.padding && attrs.padding > 0) {
                    return slice(unpacked, new Array(unpacked.shape.length).fill(0), attrs.originalShape);
                }
                return unpacked;
            },
        };
    },
};

registerGradient(packGradConfig);
