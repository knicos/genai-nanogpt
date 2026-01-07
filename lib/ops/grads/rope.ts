import RoPECache from '@base/layers/RoPECache';
import { registerGradient, GradConfig, Tensor, NamedAttrMap } from '@tensorflow/tfjs-core';
import { rope } from '../rope';

export const ropeGradConfig: GradConfig = {
    kernelName: 'Rope',
    inputsToSave: [],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], _: Tensor[], attrs: NamedAttrMap) => {
        const { ropeCache } = attrs as unknown as { ropeCache: RoPECache };

        // To invert RoPE, apply RoPE with -sin (i.e., swap sin sign)
        // This is mathematically equivalent to applying the inverse rotation

        // Use the same rope logic, but with negated sin
        const pastLen = 0; // Not used during backprop, can be set to 0
        const gradInput = rope(dy as Tensor, ropeCache, pastLen, true);

        return { x: () => gradInput };
    },
};

registerGradient(ropeGradConfig);
