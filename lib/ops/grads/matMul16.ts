import { GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';
import { matMul16 } from '../matMul16';

export const matMul16GradConfig: GradConfig = {
    kernelName: 'MatMul16',
    inputsToSave: ['A', 'B'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[], attrs: unknown) => {
        const [A, B] = saved;

        if (Array.isArray(dy)) {
            throw new Error('Expected dy to be a single Tensor');
        }

        const intDy = dy;

        const { transposeA, transposeB } = attrs as {
            transposeA: boolean;
            transposeB: boolean;
        };

        if (!transposeA && !transposeB) {
            return {
                A: () => matMul16(intDy, B, false, true),
                B: () => matMul16(A, intDy, true, false),
            };
        } else if (!transposeA && transposeB) {
            return {
                A: () => matMul16(intDy, B, false, false),
                B: () => matMul16(A, intDy, true, false),
            };
        } else if (transposeA && !transposeB) {
            return {
                A: () => matMul16(B, intDy, false, true),
                B: () => matMul16(A, intDy, false, false),
            };
        } else {
            throw new Error('Gradient for transposeA=true and transposeB=true is not supported yet.');
        }
    },
};

registerGradient(matMul16GradConfig);
