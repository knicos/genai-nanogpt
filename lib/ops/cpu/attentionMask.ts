import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    matMul,
    scalar,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';

// CPU fallback implementation
function attentionMaskCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo {
    const { q, k, mask } = args.inputs as { q: Tensor; k: Tensor; mask?: Tensor };
    const { divisor } = args.attrs as { divisor: number };

    const T1 = q.shape[2]!; // Sequence length
    const T2 = k.shape[2]!; // Sequence length

    // Causal self-attention
    const attUnscaled = matMul(q, k, false, true); // (B, nh, T1, T2)
    const att = attUnscaled.mul(scalar(divisor)); // Scale by sqrt(d_k)
    if (mask) {
        const mask2 = mask.slice([0, 0], [T1, T2]).expandDims(0).expandDims(0); // (1,1,T1,T2)
        const maskedAtt = att.add(mask2);
        return maskedAtt;
    }
    return att;
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
