import {
    concat,
    gather,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    range,
    registerKernel,
    stack,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';

export function applyRoPE(sinCache: Tensor, cosCache: Tensor, rotaryDim: number, q: Tensor, pastLen: number): Tensor {
    const hs = q.shape[3]!;
    const rd = rotaryDim;
    if (rd > hs) return q;

    const Tcur = q.shape[2]!;

    const half = rd / 2;

    // Use rank-4 tensors for WebGL compatibility (avoid 5D broadcasting)
    const cos = cosCache.slice([pastLen, 0, 0], [Tcur, half, 1]).reshape([1, 1, Tcur, half]);
    const sin = sinCache.slice([pastLen, 0, 0], [Tcur, half, 1]).reshape([1, 1, Tcur, half]);

    const B = q.shape[0]!;
    const nh = q.shape[1]!;

    const evenIdx = range(0, rd, 2, 'int32');
    const oddIdx = range(1, rd, 2, 'int32');

    const rotate = (x: Tensor) => {
        const rotPart = x.slice([0, 0, 0, 0], [B, nh, Tcur, rd]);
        const restPart = rd < hs ? x.slice([0, 0, 0, rd], [B, nh, Tcur, hs - rd]) : null;

        const even = gather(rotPart, evenIdx, 3); // [B, nh, Tcur, half]
        const odd = gather(rotPart, oddIdx, 3); // [B, nh, Tcur, half]

        const evenCost = even.mul(cos);
        const oddSin = odd.mul(sin);
        const evenRot = evenCost.sub(oddSin);
        const oddCost = odd.mul(cos);
        const evenSin = even.mul(sin);
        const oddRot = oddCost.add(evenSin);

        even.dispose();
        odd.dispose();
        cos.dispose();
        sin.dispose();
        evenCost.dispose();
        oddSin.dispose();
        oddCost.dispose();
        evenSin.dispose();

        // Interleave (even', odd') -> last dim size rd, without elementwise ops on rank-5
        const stacked = stack([evenRot, oddRot], -1); // [B, nh, Tcur, half, 2]
        evenRot.dispose();
        oddRot.dispose();
        const rotated = stacked.reshape([B, nh, Tcur, rd]); // [B, nh, Tcur, rd]
        stacked.dispose();

        return restPart ? concat([rotated, restPart], 3) : rotated;
    };

    const qR = rotate(q);

    evenIdx.dispose();
    oddIdx.dispose();

    return qR;
}

// CPU fallback implementation
export function ropeCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo {
    const { x, sin, cos } = args.inputs as { x: Tensor; sin: Tensor; cos: Tensor };
    const { pastLen } = args.attrs as { pastLen: number };

    const rotaryDim = x.shape[3]!;
    return applyRoPE(sin, cos, rotaryDim, x, pastLen);
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'Rope',
    backendName: 'cpu',
    kernelFunc: ropeCPU,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'Rope',
    backendName: 'tensorflow',
    kernelFunc: ropeCPU,
};

registerKernel(tensorflowKernelConfig);
