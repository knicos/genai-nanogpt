import { flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { createReduceInfo, createReductionShader16, reduce, ReduceWebGPUProgram } from './utils/reductions';
import {
    backend_util,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    sum,
    Tensor,
    TensorInfo,
    util,
} from '@tensorflow/tfjs-core';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { isPackedTensor } from '@base/utilities/packed';
import { PackedTensorInfo } from '@base/patches/PackedTensor';
import { transpose16 } from '../transpose16';

class SumProgram16 implements ReduceWebGPUProgram {
    outputShape: number[];
    shaderKey = 'sum16';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    variableNames = ['x'];
    uniforms = 'reduceSize : i32,';
    inputShape: number[];
    size = true;
    packed = true;
    outputComponent: number;
    variableComponents?: number[];
    keepDims: boolean;

    constructor(reduceInfo: backend_util.ReduceInfo, keepDims = true) {
        this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
        this.outputShape = [reduceInfo.batchSize / 2];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = [reduceInfo.batchSize / 2, 1, 1];
        this.outputComponent = 1;
        this.variableComponents = [1];
        this.keepDims = keepDims;
    }

    getUserCode(): string {
        const workgroupSizeX = this.workgroupSize[0];

        const outputSnippet = `result[outputIndex] = i32(pack2x16float(bestValue));`;

        return createReductionShader16(workgroupSizeX, 'sum', '', '', outputSnippet, false);
    }
}

function sum16GPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { axis, keepDims } = args.attrs as { axis?: number | number[]; keepDims?: boolean };
    const backend = args.backend as WebGPUBackend;
    const toDispose: Tensor[] = [];

    const packed = isPackedTensor(x);

    if (!packed) {
        return sum(x, axis, keepDims);
    }

    const origAxes = util.parseAxisParam(axis ?? -1, x.shape);
    let axes = origAxes;
    const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);

    let input = x;
    if (permutedAxes != null) {
        input = transpose16(x, permutedAxes);
        axes = backend_util.getInnerMostAxes(axes.length, input.shape.length);
        toDispose.push(input);
    }

    const reduceInfo = createReduceInfo([input], -1);
    const program = new SumProgram16(reduceInfo, keepDims);

    const result: PackedTensorInfo = reduce(program, [input], false, backend);
    result.packed = true;
    toDispose.forEach((t) => t.dispose());
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'Sum16',
    backendName: 'webgpu',
    kernelFunc: sum16GPU,
};

registerKernel(kernelConfig);
