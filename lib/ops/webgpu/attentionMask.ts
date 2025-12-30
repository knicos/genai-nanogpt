import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';
import { isPackedTensor } from '@base/utilities/packed';
import { matMul16 } from '../matMul16';
import AttentionMaskProgram32 from './attentionMask32_program';

function attentionMaskGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { q, k } = args.inputs as { q: Tensor; k: Tensor };
    const { divisor, pastLen } = args.attrs as { divisor: number; pastLen: number };

    const backend = args.backend as WebGPUBackend;

    const packed = isPackedTensor(q) && isPackedTensor(k);

    if (packed) {
        // Use fused matMul16 with causal mask
        return matMul16(q, k, false, true, { causalMask: true, pastLen, scale: divisor });
    }

    // TODO: Implement fused matMul for unpacked tensors as well

    const batchSize = q.shape[0];
    const T1 = q.shape[2]!; // Sequence length
    const T2 = k.shape[2]!; // Sequence length
    const nh = q.shape[1]!; // Number of heads
    const hs = q.shape[3]!; // Head size

    assertShapesMatch(k.shape, [batchSize, nh, T2, hs], 'Error in AttentionMask: ');
    if (divisor === 0) {
        throw new Error('Divisor must be non-zero in AttentionMask');
    }
    if (pastLen < 0) {
        throw new Error('pastLen must be non-negative in AttentionMask');
    }

    const program = new AttentionMaskProgram32(batchSize, nh, T1, T2, hs);
    const uniformData = [
        { type: 'float32', data: [divisor] },
        { type: 'int32', data: [pastLen] },
        { type: 'float32', data: [Number.NEGATIVE_INFINITY] },
    ];

    const dtype = q.dtype;
    return backend.runWebGPUProgram(program, [q, k], dtype, uniformData);
}

const kernelConfig: KernelConfig = {
    kernelName: 'AttentionMask',
    backendName: 'webgpu',
    kernelFunc: attentionMaskGPU,
};

registerKernel(kernelConfig);
