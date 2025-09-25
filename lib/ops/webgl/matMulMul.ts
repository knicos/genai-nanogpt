import { KernelConfig, KernelFunc, registerKernel, TensorInfo } from '@tensorflow/tfjs-core';

import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { batchMatMulGeluImpl } from './matMulGelu';

const MUL_PACKED = `
    return a * b;
`;

export function batchMatMulKernel(args: {
    inputs: { x: TensorInfo; kernel: TensorInfo; y: TensorInfo };
    attrs: { transposeA: boolean; transposeB: boolean };
    backend: MathBackendWebGL;
}) {
    const { inputs, backend, attrs } = args;
    const { x, kernel, y } = inputs;
    const { transposeA, transposeB } = attrs;

    if (x === undefined || kernel === undefined) {
        throw new Error('BatchMatMul requires two input tensors.');
    }

    return batchMatMulGeluImpl({
        a: x,
        b: kernel,
        transposeA,
        transposeB,
        backend,
        activationSnippet: MUL_PACKED,
        multiplier: y,
    });
}

const matMulMulConfig: KernelConfig = {
    kernelName: 'MatMulMul',
    backendName: 'webgl',
    kernelFunc: batchMatMulKernel as unknown as KernelFunc,
};

registerKernel(matMulMulConfig);
