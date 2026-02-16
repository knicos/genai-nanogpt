import { reduce, ReduceProgram } from './utils/reductions';
import {
    backend_util,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';

import WebGPUBackendPatch from '@base/patches/webgpu_backend';
import createDeviceInformation, { DeviceInformation } from './utils/deviceInfo';

class ClipScaleProgram extends ReduceProgram {
    shaderKey = 'clipscale';

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

        this.uniforms += 'scaling: f32, clipNorm: f32';
    }

    protected override getPreprocessSnippet(): string {
        return `
            candidate = candidate / 100.0f;
        `;
    }

    protected override getWriteSnippet(): string {
        return `
            if (tid == 0) {
                let cnorm = uniforms.clipNorm;
                result[0] = (cnorm / max(cnorm, sqrt(bestValue))) * uniforms.scaling;
            }
        `;
    }
}

function clipScaleGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { invLossScaling, clipNorm } = args.attrs as { invLossScaling: number; clipNorm: number };
    const backend = args.backend as WebGPUBackendPatch;
    const toDispose: Tensor[] = [];

    if (x.shape.length !== 1) {
        throw new Error(`ClipScaleProgram requires 1D input, but got shape ${x.shape}`);
    }

    const deviceInfo = createDeviceInformation(backend);
    let workgroupSize = 128;
    let workPerThread = 1;

    // Find best workgroup size that divides the reduction size and is less than the max
    const reduceSize = x.shape[0];

    if (reduceSize <= 16) {
        workgroupSize = 16;
    } else if (reduceSize <= 32) {
        workgroupSize = 32;
    } else if (reduceSize <= 64) {
        workgroupSize = 64;
    }
    if (reduceSize > 128) {
        workPerThread = Math.ceil(reduceSize / 128);
    }

    const reduceInfo: backend_util.ReduceInfo = {
        inSize: workgroupSize * workPerThread,
        outSize: 1,
        batchSize: 1,
        windowSize: workgroupSize,
    };
    const program = new ClipScaleProgram(deviceInfo, reduceInfo, workgroupSize);

    const result = reduce(program, [x], backend, [
        { type: 'float32', data: [invLossScaling] },
        { type: 'float32', data: [clipNorm] },
    ]);
    toDispose.forEach((t) => t.dispose());
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'ClipScale',
    backendName: 'webgpu',
    kernelFunc: clipScaleGPU,
};

registerKernel(kernelConfig);
