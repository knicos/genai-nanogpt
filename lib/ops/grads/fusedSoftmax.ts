import { GradConfig, mul, NamedAttrMap, registerGradient, SoftmaxAttrs, sub, sum, Tensor } from '@tensorflow/tfjs-core';
import { mulDrop } from '../mulDrop';

interface FusedSoftmaxAttrs extends SoftmaxAttrs {
    dropoutRate?: number;
    seed?: number;
}

export const softmaxGradConfig: GradConfig = {
    kernelName: 'FusedSoftmax',
    outputsToSave: [true],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[], attrs: NamedAttrMap) => {
        const [y] = saved;
        const { dim, dropoutRate, seed } = attrs as unknown as FusedSoftmaxAttrs;
        const keepDims = true;

        // Mul dy by dropout mask if dropout was applied
        const dyTimesY = dropoutRate && seed ? mulDrop(dy as Tensor, y, dropoutRate, seed) : mul(dy as Tensor, y);
        return {
            logits: () => {
                const sumDyTimesY = sum(dyTimesY, [dim], keepDims);
                const sumMulYTimesY = mul(sumDyTimesY, y);
                sumDyTimesY.dispose();
                const result = sub(dyTimesY, sumMulYTimesY);
                sumMulYTimesY.dispose();
                return result;
            },
        };
    },
};

registerGradient(softmaxGradConfig);
