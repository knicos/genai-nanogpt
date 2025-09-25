import { NamedAttrMap, registerGradient, GradConfig, Tensor, scalar } from '@tensorflow/tfjs-core';
import { matMulMul } from '../matMulMul';

const attentionMaskGradConfig: GradConfig = {
    kernelName: 'AttentionMask',
    inputsToSave: ['q', 'k'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[], attrs: NamedAttrMap) => {
        if (Array.isArray(dy)) {
            throw new Error('Expected dy to be a single Tensor');
        }
        const [q, k] = saved as Tensor[];
        const { divisor } = attrs as { divisor: number };

        return {
            q: () => matMulMul(dy, k, scalar(divisor)),
            k: () => {
                const qt = q.transpose([0, 1, 3, 2]);
                const result = matMulMul(qt, dy, scalar(divisor));
                qt.dispose();
                return result.transpose([0, 1, 3, 2]);
            },
            mask: () => dy,
            divisor: () => {
                const attUnscaled = q.matMul(k, false, true);
                const dyAtt = dy.mul(attUnscaled);
                attUnscaled.dispose();
                return dyAtt.sum();
            },
        };
    },
};

registerGradient(attentionMaskGradConfig);
