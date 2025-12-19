import { isPackedTensor, packTensor } from '@base/utilities/packed';
import { registerGradient, GradConfig, engine, Tensor } from '@tensorflow/tfjs-core';

function rope(x: Tensor, sinCache: Tensor, cosCache: Tensor, pastLength: number): Tensor {
    return engine().runKernel('Rope', { x, sin: sinCache, cos: cosCache }, { pastLen: pastLength });
}

export const ropeGradConfig: GradConfig = {
    kernelName: 'Rope',
    inputsToSave: ['sin', 'cos'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        const [sin, cos] = saved;

        // To invert RoPE, apply RoPE with -sin (i.e., swap sin sign)
        // This is mathematically equivalent to applying the inverse rotation

        // Negate sin cache
        const negSin = sin.neg();

        const ispacked = isPackedTensor(dy as Tensor);

        // Use the same rope logic, but with negated sin
        const pastLen = 0; // Not used during backprop, can be set to 0
        const gradInput = rope(dy as Tensor, negSin, cos, pastLen);

        negSin.dispose();

        return { x: () => (ispacked ? packTensor(gradInput) : gradInput) };
    },
};

registerGradient(ropeGradConfig);
