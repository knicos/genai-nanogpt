import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import {
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { BinaryOpProgram, BinaryOpScalarProgram, BinaryOpType } from './utils/binary_op';

function mul16GPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { a, b } = args.inputs as { a: Tensor; b: Tensor };
    const backend = args.backend as WebGPUBackend;

    const program =
        b.shape.length === 0
            ? new BinaryOpScalarProgram(BinaryOpType.MUL, a.shape)
            : new BinaryOpProgram(BinaryOpType.MUL, a.shape, b.shape);

    const result = backend.runWebGPUProgram(program, [a, b], 'packedF16');
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'Mul16',
    backendName: 'webgpu',
    kernelFunc: mul16GPU,
};

registerKernel(kernelConfig);
