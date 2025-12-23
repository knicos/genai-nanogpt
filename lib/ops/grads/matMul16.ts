import { GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';
import { matMul16, matMul16ScaleA, matMul16ScaleB } from '../matMul16';
import { dGelu } from '../gelu';

export const matMul16GradConfig: GradConfig = {
    kernelName: 'MatMul16',
    inputsToSave: ['A', 'B'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[], attrs: unknown) => {
        const [A, B] = saved;

        if (Array.isArray(dy)) {
            throw new Error('Expected dy to be a single Tensor');
        }

        let intDy = dy;

        const { transposeA, transposeB, scale, activation } = attrs as {
            transposeA: boolean;
            transposeB: boolean;
            scale?: number;
            activation?: 'gelu';
        };

        if (activation === 'gelu') {
            const oldDy = intDy;
            const inputMatMul = matMul16(A, B, transposeA, transposeB);
            intDy = dGelu(oldDy, inputMatMul);
            oldDy.dispose();
            inputMatMul.dispose();
        }

        if (!transposeA && !transposeB) {
            return {
                A: () =>
                    scale !== undefined
                        ? matMul16ScaleA(intDy, B, scale, false, true)
                        : matMul16(intDy, B, false, true),
                B: () =>
                    scale !== undefined
                        ? matMul16ScaleB(A, intDy, scale, true, false)
                        : matMul16(A, intDy, true, false),
            };
        } else if (!transposeA && transposeB) {
            return {
                A: () =>
                    scale !== undefined
                        ? matMul16ScaleA(intDy, B, scale, false, false)
                        : matMul16(intDy, B, false, false),
                B: () =>
                    scale !== undefined
                        ? matMul16ScaleB(A, intDy, scale, true, false)
                        : matMul16(A, intDy, true, false),
            };
        } else if (transposeA && !transposeB) {
            return {
                A: () =>
                    scale !== undefined
                        ? matMul16ScaleB(B, intDy, scale, false, true)
                        : matMul16(B, intDy, false, true),
                B: () =>
                    scale !== undefined
                        ? matMul16ScaleB(A, intDy, scale, false, false)
                        : matMul16(A, intDy, false, false),
            };
        } else {
            throw new Error('Gradient for transposeA=true and transposeB=true is not supported yet.');
        }
    },
};

registerGradient(matMul16GradConfig);
