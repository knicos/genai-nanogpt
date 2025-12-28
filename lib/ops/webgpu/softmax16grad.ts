import {
    backend_util,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { createReduceInfo, createReductionShader16, reduce, ReduceWebGPUProgram } from './utils/reductions';
import { flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { isPackedTensor } from '@base/utilities/packed';
import { PackedTensorInfo } from '@base/patches/PackedTensor';

class SoftmaxGradProgram16 implements ReduceWebGPUProgram {
    outputShape: number[];
    shaderKey = 'Softmax16Grad';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    variableNames = ['dy', 'softmaxOutput'];
    uniforms = 'reduceSize : i32,';
    inputShape: number[];
    size = true;
    packed = true;

    constructor(reduceInfo: backend_util.ReduceInfo) {
        this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
        this.outputShape = this.inputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = [reduceInfo.batchSize, 1, 1];
    }

    getUserCode(): string {
        const workgroupSizeX = this.workgroupSize[0];

        const inputReadSnippet = `
            let d: vec2<f32> = unpack2x16float(u32(dy[index]));
            let l: vec2<f32> = unpack2x16float(u32(softmaxOutput[index]));
            return d * l;
        `;

        const inputSnippet = '';

        const reducedSnippet = '';

        const outputSnippet = `
            let d: vec2<f32> = unpack2x16float(u32(dy[offset + k]));
            let l: vec2<f32> = unpack2x16float(u32(softmaxOutput[offset + k]));
            let outVal = l * (d - bestValue);
            result[offset + k] = i32(pack2x16float(outVal));
        `;

        return createReductionShader16(
            workgroupSizeX,
            'sum',
            inputSnippet,
            reducedSnippet,
            outputSnippet,
            inputReadSnippet
        );
    }
}

function softmaxGradGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { dy, softmaxOutput } = args.inputs as { dy: Tensor; softmaxOutput: Tensor };
    const backend = args.backend as WebGPUBackend;

    const packedDY = isPackedTensor(dy);
    const packedSoftmaxOutput = isPackedTensor(softmaxOutput);
    const packed = packedDY && packedSoftmaxOutput;

    if (!packed) {
        throw new Error('softmaxGradGPU: dy and softmaxOutput must be packed tensors');
    }

    const inputs = [dy, softmaxOutput];
    const reduceInfo = createReduceInfo(inputs, -1);
    const program = new SoftmaxGradProgram16(reduceInfo);

    const result: PackedTensorInfo = reduce(program, inputs, true, backend);
    result.packed = packed;
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'Softmax16Grad',
    backendName: 'webgpu',
    kernelFunc: softmaxGradGPU,
};

registerKernel(kernelConfig);
