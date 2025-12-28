import { registerGradient, GradConfig, Tensor, squeeze } from '@tensorflow/tfjs-core';
import { matMul16GradConfig } from './matMul16';
import { concat16 } from '../concat16';
import { sum16 } from '../sum16';
import { NamedGradientMap } from '@tensorflow/tfjs-core/dist/tape';
import { isPackedTensor, packTensor } from '@base/utilities/packed';

const qkvGradConfig: GradConfig = {
    kernelName: 'QKV',
    inputsToSave: ['x', 'kernel'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        // dy: [dq, dk, dv]
        // x: input tensor
        // kernel: fused kernel weights
        const [dq, dk, dv] = dy as Tensor[];
        const [x] = saved;

        // Concatenate along head axis (axis=1): [B, heads, T, hs] * 3 -> [B, 3*heads, T, hs]
        const dQKV = concat16([dq, dk, dv], 1);

        // Probably ok, but check it
        dq.dispose();
        dk.dispose();
        dv.dispose();

        const originalShape = [x.shape[0], x.shape[1]!, 3 * x.shape[2]!];

        // Call matMul16's gradient
        const grads = matMul16GradConfig.gradFunc(dQKV, saved, {
            transposeA: false,
            transposeB: false,
            originalShape,
            perm: [0, 2, 1, 3],
        });

        dQKV.dispose();

        // grads: { x: Tensor, kernel: Tensor }
        return {
            x: () => grads.A(),
            kernel: () => {
                const bGrad = grads.B();
                const sumBgrad = bGrad.shape[0] === 1 ? squeeze(bGrad, [0]) : sum16(bGrad, 0);
                bGrad.dispose();
                return isPackedTensor(bGrad) ? packTensor(sumBgrad) : sumBgrad;
            },
        };
    },
};

export function qkvGrad(dy: Tensor[], x: Tensor, kernel: Tensor): NamedGradientMap {
    return qkvGradConfig.gradFunc(dy, [x, kernel], {});
}

registerGradient(qkvGradConfig);
