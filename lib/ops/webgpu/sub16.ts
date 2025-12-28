import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import {
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { BinaryOpProgram, BinaryOpType } from './utils/binary_op';
import { PackedTensorInfo } from '@base/patches/PackedTensor';

function sub16GPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { a, b } = args.inputs as { a: Tensor; b: Tensor };
    const backend = args.backend as WebGPUBackend;

    const program = new BinaryOpProgram(BinaryOpType.SUB, a.shape, b.shape);

    const result: PackedTensorInfo = backend.runWebGPUProgram(program, [a, b], 'int32');
    result.packed = true;
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'Sub16',
    backendName: 'webgpu',
    kernelFunc: sub16GPU,
};

registerKernel(kernelConfig);
