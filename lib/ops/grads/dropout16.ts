import { GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';
import { dropout16 } from '../dropout16';

const dropout16GradConfig: GradConfig = {
    kernelName: 'Dropout16',
    inputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], _: Tensor[], attrs: unknown) => {
        const { dropout, seed } = attrs as { dropout: number; seed?: number };
        const derX = () => {
            return dropout16(dy as Tensor, dropout, seed);
        };

        return { x: derX };
    },
};

registerGradient(dropout16GradConfig);
