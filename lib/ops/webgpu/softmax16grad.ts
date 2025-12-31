import {
    backend_util,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { createReduceInfo, reduce, ReduceProgram } from './utils/reductions';
import { isPackedTensor } from '@base/utilities/packed';
import WebGPUBackendPatch from '@base/patches/webgpu_backend';
import createDeviceInformation, { DeviceInformation } from './utils/deviceInfo';

class SoftmaxGradProgram16 extends ReduceProgram {
    constructor(deviceInfo: DeviceInformation, reduceInfo: backend_util.ReduceInfo) {
        super(deviceInfo, reduceInfo, { reductionOp: 'sum', elementwise: true }, true);
        this.shaderKey = 'SoftmaxGrad16';
        this.variableNames = ['dy', 'softmaxOutput'];
        this.variableComponents = [1, 1];
    }

    protected override getReadSnippet(): string {
        return `
            let d: vec2<f32> = unpack2x16float(u32(dy[index]));
            let l: vec2<f32> = unpack2x16float(u32(softmaxOutput[index]));
            return d * l;
        `;
    }

    protected override getWriteSnippet(): string {
        return `
            let d: vec2<f32> = unpack2x16float(u32(dy[offset + k]));
            let l: vec2<f32> = unpack2x16float(u32(softmaxOutput[offset + k]));
            let outVal = l * (d - bestValue);
            result[offset + k] = i32(pack2x16float(outVal));
        `;
    }
}

function softmaxGradGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { dy, softmaxOutput } = args.inputs as { dy: Tensor; softmaxOutput: Tensor };
    const backend = args.backend as WebGPUBackendPatch;

    const deviceInfo = createDeviceInformation(backend);

    const packedDY = isPackedTensor(dy);
    const packedSoftmaxOutput = isPackedTensor(softmaxOutput);
    const packed = packedDY && packedSoftmaxOutput;

    if (!packed) {
        throw new Error('softmaxGradGPU: dy and softmaxOutput must be packed tensors');
    }

    const inputs = [dy, softmaxOutput];
    const reduceInfo = createReduceInfo(inputs, -1);
    const program = new SoftmaxGradProgram16(deviceInfo, reduceInfo);

    const result = reduce(program, inputs, backend);
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'Softmax16Grad',
    backendName: 'webgpu',
    kernelFunc: softmaxGradGPU,
};

registerKernel(kernelConfig);
