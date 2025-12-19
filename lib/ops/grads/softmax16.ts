import { GradConfig, mul, NamedAttrMap, registerGradient, SoftmaxAttrs, sub, sum, Tensor } from '@tensorflow/tfjs-core';
import { mulDrop } from '../mulDrop';
import { unpack16 } from '../unpack16';
import { pack16 } from '../pack16';
import { isPackedTensor } from '@base/utilities/packed';

interface FusedSoftmaxAttrs extends SoftmaxAttrs {
    dropoutRate?: number;
    seed?: number;
}

export const softmax16GradConfig: GradConfig = {
    kernelName: 'Softmax16',
    outputsToSave: [true],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[], attrs: NamedAttrMap) => {
        const [y] = saved;
        const { dim, dropoutRate, seed } = attrs as unknown as FusedSoftmaxAttrs;
        const keepDims = true;

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

        const dyUnpacked = unpack16(dy as Tensor);
        dy.dispose();
        const yUnpacked = unpack16(y);
        y.dispose();

        // Mul dy by dropout mask if dropout was applied
        const dyTimesY =
            dropoutRate && seed ? mulDrop(dyUnpacked, yUnpacked, dropoutRate, seed) : mul(dyUnpacked, yUnpacked);
        return {
            logits: () => {
                const sumDyTimesY = sum(dyTimesY, [dim], keepDims);
                const sumMulYTimesY = mul(sumDyTimesY, yUnpacked);
                sumDyTimesY.dispose();
                const result = sub(dyTimesY, sumMulYTimesY);
                sumMulYTimesY.dispose();
                const packed = pack16(result);
                result.dispose();
                return packed;
            },
        };
    },
};

registerGradient(softmax16GradConfig);
