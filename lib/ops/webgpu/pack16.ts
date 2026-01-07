import type { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';
import PackProgram from './pack16_program';

function packGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { scaling, padding } = args.attrs as { scaling: number; padding: number; originalShape?: number[] };
    const backend = args.backend as WebGPUBackend;

    if (x.shape[x.shape.length - 1] % 2 !== 0) {
        throw new Error('Last dimension of input tensor must be even to use Pack16.');
    }

    if (args.attrs) {
        args.attrs.originalShape = x.shape;
    }

    const program = new PackProgram(x.shape, padding);

    const hasScaling = scaling !== 1.0;
    if (hasScaling) {
        program.useScaling();
    }

    const uniformData = [{ type: 'float32', data: [scaling] }];

    return backend.runWebGPUProgram(program, [x], 'packedF16', hasScaling ? uniformData : undefined);
}

const kernelConfig: KernelConfig = {
    kernelName: 'Pack16',
    backendName: 'webgpu',
    kernelFunc: packGPU,
};

registerKernel(kernelConfig);
