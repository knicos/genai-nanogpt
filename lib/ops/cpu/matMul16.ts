import { isPackedTensor } from '@base/utilities/packed';
import {
    KernelConfig,
    matMul,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';

function matMul16CPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { A, B } = args.inputs as { A: Tensor; B: Tensor };
    const { transposeA, transposeB } = args.attrs as { transposeA: boolean; transposeB: boolean };

    const wasAUnpacked = !isPackedTensor(A);
    const wasBUnpacked = !isPackedTensor(B);

    if (wasAUnpacked && wasBUnpacked) {
        return matMul(A, B, transposeA, transposeB);
    }

    throw new Error('MatMul16 CPU kernel only supports packed tensors currently.');
}

const kernelConfig: KernelConfig = {
    kernelName: 'MatMul16',
    backendName: 'cpu',
    kernelFunc: matMul16CPU,
};

registerKernel(kernelConfig);
