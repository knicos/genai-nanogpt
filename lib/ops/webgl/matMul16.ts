import { isPackedTensor } from '@base/utilities/packed';
import {
    KernelConfig,
    matMul,
    mul,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    reshape,
    scalar,
    Tensor,
    TensorInfo,
    transpose,
} from '@tensorflow/tfjs-core';
import { matMulMul } from '../matMulMul';
import { matMulGelu } from '../matMulGelu';

function matMul16GPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { A, B } = args.inputs as { A: Tensor; B: Tensor };
    const { transposeA, transposeB, scale, activation, scaleA, scaleB, forceOutputShape, perm } = args.attrs as {
        transposeA: boolean;
        transposeB: boolean;
        scale?: number;
        scaleA?: number;
        scaleB?: number;
        activation?: 'gelu';
        forceOutputShape?: number[];
        perm?: number[];
        originalShape?: number[];
    };

    const wasAUnpacked = !isPackedTensor(A);
    const wasBUnpacked = !isPackedTensor(B);

    // Unpacked 32-bit float version uses standard matMul
    // This also adds the optional features but without fusion
    if (wasAUnpacked && wasBUnpacked) {
        const sA = scaleA !== undefined ? mul(A, scalar(scaleA)) : A;
        const sB = scaleB !== undefined ? mul(B, scalar(scaleB)) : B;

        let result: Tensor;
        if (scale !== undefined) {
            result = matMulMul(sA, sB, scalar(scale), transposeA, transposeB);
        } else if (activation === 'gelu') {
            result = matMulGelu(sA, sB);
        } else {
            result = matMul(sA, sB, transposeA, transposeB);
        }

        // Apply forced output shape and perm if needed
        if (perm) {
            if (forceOutputShape) {
                const reshaped = reshape(result, forceOutputShape);
                result.dispose();
                const permuted = transpose(reshaped, perm);
                reshaped.dispose();
                return permuted;
            } else {
                const permuted = transpose(result, perm);
                result.dispose();
                return permuted;
            }
        } else if (forceOutputShape) {
            const r = reshape(result, forceOutputShape);
            result.dispose();
            return r;
        } else {
            return result;
        }
    }

    throw new Error('Not implemented: matMul16 with packed inputs');
}

const kernelConfig: KernelConfig = {
    kernelName: 'MatMul16',
    backendName: 'webgl',
    kernelFunc: matMul16GPU,
};

registerKernel(kernelConfig);
