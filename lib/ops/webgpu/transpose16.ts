import { isPackedTensor } from '@base/utilities/packed';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import {
    engine,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
    transpose,
    TransposeAttrs,
} from '@tensorflow/tfjs-core';

import { reshape16 } from '../reshape16';
import TransposeSharedProgram16 from './transpose16_shared_program';
import TransposeProgram16 from './transpose16_program';

function transpose16_(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { inputs, attrs } = args;
    const { x } = inputs as { x: Tensor };
    const { perm } = attrs as unknown as TransposeAttrs;

    const backend = args.backend as WebGPUBackend;

    const packed = isPackedTensor(x);

    if (packed && perm[perm.length - 1] !== x.shape.length - 1) {
        const rank = x.shape.length;

        // For rank 4 tensors, we reshape to rank 3, transpose, then reshape back to rank 4
        const newPerm = rank === 4 ? perm.map((p) => p - 1).filter((p) => p >= 0) : perm;
        const reshaped = rank === 4 ? reshape16(x, [x.shape[0] * x.shape[1]!, x.shape[2]!, x.shape[3]!]) : x;

        // Accepts rank 2 or 3, where 3 has a batch dimension
        const program = new TransposeSharedProgram16(reshaped.shape, newPerm);
        const output = backend.runWebGPUProgram(program, [reshaped], 'packedF16');

        // If rank 4, reshape back to rank 4
        if (rank === 4) {
            reshaped.dispose();
            const outputTensor = engine().makeTensorFromTensorInfo(output);
            const reshapedOutput = reshape16(outputTensor, [
                x.shape[0],
                x.shape[1]!,
                output.shape[1]!,
                output.shape[2]!,
            ]);
            outputTensor.dispose();
            return reshapedOutput;
        }
        return output;
    }

    // If the last dimension is not moved, we can use the regular transpose program
    if (packed) {
        const program = new TransposeProgram16(x.shape, perm);
        const output = backend.runWebGPUProgram(program, [x], 'packedF16');
        return output;
    } else {
        return transpose(x, perm);
    }
}

const webgpuConfig: KernelConfig = {
    kernelName: 'Transpose16',
    backendName: 'webgpu',
    kernelFunc: transpose16_,
};
registerKernel(webgpuConfig);
