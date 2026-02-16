import { reduce, ReduceProgram } from './utils/reductions';
import {
    backend_util,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
    util,
} from '@tensorflow/tfjs-core';

import WebGPUBackendPatch from '@base/patches/webgpu_backend';
import createDeviceInformation, { DeviceInformation } from './utils/deviceInfo';

class Norm2Program extends ReduceProgram {
    shaderKey = 'norm2';
    atomic = true;

    constructor(deviceInfo: DeviceInformation, reduceInfo: backend_util.ReduceInfo, workgroupSize: number) {
        super(
            deviceInfo,
            reduceInfo,
            {
                reductionOp: 'sum',
                elementwise: false,
                forceWorkgroupSize: workgroupSize,
            },
            false
        );

        this.uniforms += 'lossScaling: f32, index: i32';
    }

    protected override getPreprocessSnippet(): string {
        return `
            candidate = candidate * uniforms.lossScaling;
            candidate = candidate * candidate;
        `;
    }

    protected override getWriteSnippet(): string {
        return `
            if (tid == 0) {
                atomicAdd(&result[uniforms.index], i32(bestValue * 100.0f));
            }
        `;
    }
}

function norm2GPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x, output } = args.inputs as { x: Tensor; output: Tensor };
    const { invLossScaling, index } = args.attrs as { invLossScaling: number; index: number };
    const backend = args.backend as WebGPUBackendPatch;
    const toDispose: Tensor[] = [];

    const deviceInfo = createDeviceInformation(backend);
    let workgroupSize = Math.min(512, backend.device.limits.maxComputeWorkgroupSizeX);
    const workPerThread = 4;

    // Find best workgroup size that divides the reduction size and is less than the max
    const reduceSize = util.sizeFromShape(x.shape);
    while (reduceSize % (workgroupSize * workPerThread) !== 0 && workgroupSize > 1) {
        workgroupSize /= 2;
    }
    if (workgroupSize === 1) {
        throw new Error(`Cannot find suitable workgroup size for Norm2Program with reduce size ${reduceSize}`);
    }

    const reduceInfo: backend_util.ReduceInfo = {
        inSize: workgroupSize * workPerThread,
        outSize: 1,
        batchSize: reduceSize / (workgroupSize * workPerThread),
        windowSize: workgroupSize,
    };
    const program = new Norm2Program(deviceInfo, reduceInfo, workgroupSize);

    const result = reduce(
        program,
        [x],
        backend,
        [
            { type: 'float32', data: [invLossScaling] },
            { type: 'int32', data: [index] },
        ],
        output
    );
    toDispose.forEach((t) => t.dispose());
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'Norm2',
    backendName: 'webgpu',
    kernelFunc: norm2GPU,
};

registerKernel(kernelConfig);
