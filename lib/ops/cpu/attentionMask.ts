import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    matMul,
    scalar,
    NamedAttrMap,
    Tensor,
    ones,
    zeros,
    linalg,
    fill,
    where,
} from '@tensorflow/tfjs-core';

// CPU fallback implementation
function attentionMaskCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo {
    const { q, k } = args.inputs as { q: Tensor; k: Tensor };
    const { divisor } = args.attrs as { divisor: number };

    const T1 = q.shape[2]!; // Sequence length
    const T2 = k.shape[2]!; // Sequence length

    // Causal mask to ensure that attention is only applied to the left in the input sequence
    const bias = linalg.bandPart(ones([T2, T2]), -1, 0).cast('bool');
    const zero = zeros([T2, T2]);
    // It must be negative infinity for softmax to ignore these positions
    // Using any other number results in small but non-zero attention weights
    // Which leaks information from the future
    const negInf = fill([T2, T2], Number.NEGATIVE_INFINITY);
    const mask = where(bias as Tensor, zero, negInf);

    // Causal self-attention
    const attUnscaled = matMul(q, k, false, true); // (B, nh, T1, T2)
    const att = attUnscaled.mul(scalar(divisor)); // Scale by sqrt(d_k)

    const mask2 = mask.slice([0, 0], [T1, T2]).expandDims(0).expandDims(0); // (1,1,T1,T2)
    const maskedAtt = att.add(mask2);
    return maskedAtt;
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'AttentionMask',
    backendName: 'cpu',
    kernelFunc: attentionMaskCPU,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'AttentionMask',
    backendName: 'tensorflow',
    kernelFunc: attentionMaskCPU,
};

registerKernel(tensorflowKernelConfig);
