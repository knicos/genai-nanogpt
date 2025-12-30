import {
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';
import { matMul16 } from '../matMul16';
import { slice16 } from '../slice16';
import { isPackedTensor } from '@base/utilities/packed';

function qkvGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo[] {
    const { x, kernel } = args.inputs as { x: Tensor; kernel: Tensor };
    const { heads } = args.attrs as { heads: number };
    const batchSize = x.shape[0];
    const seqLength = x.shape[1]!;
    const C = x.shape[2]!;

    const packed = isPackedTensor(x);

    assertShapesMatch(kernel.shape, [packed ? C * 2 : C, 3 * C], 'Error in QKV: ');
    if (C % heads !== 0) {
        throw new Error(`Channel dimension ${C} must be divisible by number of heads ${heads} in QKV.`);
    }

    // TODO: I wonder if we can fuse the slicing here to avoid creating the full qkvMat
    // That would require multiple outputs from matMul16 though, which is not currently supported
    const qkvMat = matMul16(x, kernel, false, false, {
        forceOutputShape: [batchSize, seqLength, 3 * heads, C / heads],
        perm: [0, 2, 1, 3],
    }); // [B, 3*nh, T, hs]

    const result = [
        slice16(qkvMat, [0, 0, 0, 0], [batchSize, heads, seqLength, C / heads]),
        slice16(qkvMat, [0, heads, 0, 0], [batchSize, heads, seqLength, C / heads]),
        slice16(qkvMat, [0, 2 * heads, 0, 0], [batchSize, heads, seqLength, C / heads]),
    ];

    qkvMat.dispose();

    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'QKV',
    backendName: 'webgpu',
    kernelFunc: qkvGPU,
};

registerKernel(kernelConfig);
