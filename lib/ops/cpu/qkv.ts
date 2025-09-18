import {
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    reshape,
    split,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';

// CPU fallback implementation
export function qkvCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo[] {
    const { x, kernel } = args.inputs as { x: Tensor; kernel: Tensor };
    const { heads } = args.attrs as { heads: number };

    const [B, T, C] = x.shape; // batch size, sequence length, embedding dimensionality

    // Calculate query, key, values for all heads in batch and move head forward to be the batch dim
    const x2d = reshape(x, [B * T, C]);
    const qkvFlat = x2d.dot(kernel); //this.cAttn.apply(x) as TF.Tensor; // (B, T, 3*C)
    //x.dispose();
    x2d.dispose();
    const qkv = reshape(qkvFlat, [B, T, 3 * C]);
    qkvFlat.dispose();

    const [q, k, v] = split(qkv, 3, -1); // Each is (B, T, C)
    qkv.dispose();

    // Reshape for multi-head attention
    const headDim = C / heads;

    const qReshaped = reshape(q, [B, T, heads, headDim]);
    q.dispose();
    const qT = qReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
    qReshaped.dispose();

    const kReshaped = reshape(k, [B, T, heads, headDim]);
    k.dispose();
    const kT = kReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
    kReshaped.dispose();

    const vReshaped = reshape(v, [B, T, heads, headDim]);
    v.dispose();
    const vT = vReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
    vReshaped.dispose();

    return [qT, kT, vT];
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'QKV',
    backendName: 'cpu',
    kernelFunc: qkvCPU,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'QKV',
    backendName: 'tensorflow',
    kernelFunc: qkvCPU,
};

registerKernel(tensorflowKernelConfig);
