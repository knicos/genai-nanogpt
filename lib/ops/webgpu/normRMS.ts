import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
// import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';

import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
    backend_util,
} from '@tensorflow/tfjs-core';

import { createReduceInfo, createReductionShader, reduce, ReduceWebGPUProgram } from './utils/reductions';

class RMSProgram implements ReduceWebGPUProgram {
    outputShape: number[];
    shaderKey = 'RMSNorm';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    variableNames = ['x', 'gamma'];
    uniforms = 'reduceSize : i32,';
    inputShape: number[];
    size = true;

    constructor(reduceInfo: backend_util.ReduceInfo) {
        this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
        this.outputShape = this.inputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = [reduceInfo.batchSize, 1, 1];
    }

    getUserCode(): string {
        const workgroupSizeX = this.workgroupSize[0];

        const inputSnippet = `
            candidate = candidate * candidate;
        `;

        const reducedSnippet = `
            bestValue = inverseSqrt(bestValue + 1e-8);
        `;

        const outputSnippet = `
            let X = f32(x[offset + k]);
            let gamma = gamma[k];
            let normalized = X * bestValue;
            let outVal = normalized * gamma;
            result[offset + k] = f32(outVal);
        `;

        return createReductionShader(workgroupSizeX, 'mean', inputSnippet, reducedSnippet, outputSnippet);
    }
}

function rmsNormGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x, gamma } = args.inputs as { x: Tensor; gamma: Tensor };
    const backend = args.backend as WebGPUBackend;

    const inputs = [x, gamma];
    const reduceInfo = createReduceInfo(inputs, -1);
    const program = new RMSProgram(reduceInfo);

    return reduce(program, inputs, backend);
}

const kernelConfig: KernelConfig = {
    kernelName: 'RMSNorm',
    backendName: 'webgpu',
    kernelFunc: rmsNormGPU,
};

registerKernel(kernelConfig);
