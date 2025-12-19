import { NamedAttrMap, registerGradient, GradConfig, Tensor } from '@tensorflow/tfjs-core';
import { matMul16Scaled } from '../matMul16';
import { transpose16 } from '../transpose16';

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
            q: () => {
                const muled = matMul16Scaled(dy, k, divisor);
                return muled;
            },
            k: () => {
                const result = matMul16Scaled(q, dy, divisor, true, false);
                const resultT = transpose16(result, [0, 1, 3, 2]);
                result.dispose();
                return resultT;
            },
        };
    },
};

registerGradient(attentionMaskGradConfig);
