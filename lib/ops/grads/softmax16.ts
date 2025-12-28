import { engine, GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';
import { isPackedTensor } from '@base/utilities/packed';

function dSoftmax16(dy: Tensor, softmaxOutput: Tensor): Tensor {
    return engine().runKernel('Softmax16Grad', { dy, softmaxOutput });
}

export const softmax16GradConfig: GradConfig = {
    kernelName: 'Softmax16',
    outputsToSave: [true],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        const [y] = saved;

        if (Array.isArray(dy)) {
            throw new Error('Expected dy to be a single Tensor');
        }

        if (!isPackedTensor(y)) {
            console.error(y);
            throw new Error('Softmax16 gradient requires packed y Tensor');
        }
        if (!isPackedTensor(dy)) {
            throw new Error('Softmax16 gradient requires packed dy Tensor');
        }

        return {
            logits: () => {
                const result = dSoftmax16(dy, y);
                return result;
            },
        };
    },
};

registerGradient(softmax16GradConfig);
