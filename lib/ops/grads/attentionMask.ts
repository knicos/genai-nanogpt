import { NamedAttrMap, registerGradient, GradConfig, Tensor } from '@tensorflow/tfjs-core';

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
            q: () => dy.matMul(k).mul(divisor),
            k: () => q.transpose([0, 1, 3, 2]).matMul(dy).mul(divisor).transpose([0, 1, 3, 2]),
            mask: () => dy,
            divisor: () => {
                const attUnscaled = q.matMul(k, false, true);
                return dy.mul(attUnscaled).sum();
            },
        };
    },
};

registerGradient(attentionMaskGradConfig);
