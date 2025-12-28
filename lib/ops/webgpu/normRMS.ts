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

import {
    createReduceInfo,
    createReductionShader16,
    createReductionShader32,
    reduce,
    ReduceWebGPUProgram,
} from './utils/reductions';
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';
import { isPackedTensor } from '@base/utilities/packed';
import { pack16 } from '../pack16';
import { PackedTensorInfo } from '@base/patches/PackedTensor';

class RMSProgram32 implements ReduceWebGPUProgram {
    outputShape: number[];
    shaderKey = 'RMSNorm';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    variableNames = ['x', 'gamma'];
    uniforms = 'reduceSize : i32,';
    inputShape: number[];
    size = true;
    packed = false;

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

        return createReductionShader32(workgroupSizeX, 'mean', inputSnippet, reducedSnippet, outputSnippet);
    }
}

class RMSProgram16 implements ReduceWebGPUProgram {
    outputShape: number[];
    shaderKey = 'RMSNorm';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    variableNames = ['x', 'gamma'];
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

        const inputSnippet = `
            candidate = candidate * candidate;
        `;

        const reducedSnippet = `
            bestValue = inverseSqrt(bestValue + 1e-8);
        `;

        const outputSnippet = `
            let X = unpack2x16float(u32(x[offset + k]));
            let gamma = unpack2x16float(u32(gamma[k]));
            let normalized = X * bestValue;
            let outVal = normalized * gamma;
            result[offset + k] = i32(pack2x16float(outVal));
        `;

        return createReductionShader16(workgroupSizeX, 'mean', inputSnippet, reducedSnippet, outputSnippet);
    }
}

function rmsNormGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x, gamma } = args.inputs as { x: Tensor; gamma: Tensor };
    const backend = args.backend as WebGPUBackend;

    const packedX = isPackedTensor(x);
    const packedGamma = isPackedTensor(gamma);
    const packed = packedX || packedGamma;

    const pX = !packed || packedX ? x : pack16(x);
    const pGamma = !packed || packedGamma ? gamma : pack16(gamma);

    const inputs = [pX, pGamma];
    const reduceInfo = createReduceInfo(inputs, -1);
    const program = packed ? new RMSProgram16(reduceInfo) : new RMSProgram32(reduceInfo);

    assertShapesMatch(pGamma.shape, [pX.shape[pX.shape.length - 1]], 'Error in RMSNorm: ');
    if (x.shape.length !== 3) {
        throw new Error(`rmsNormGPU: input rank ${x.shape.length} not supported, only rank 3 is supported`);
    }
    if (reduceInfo.inSize !== pX.shape[pX.shape.length - 1]) {
        throw new Error(
            `rmsNormGPU: reduction size ${reduceInfo.inSize} does not match expected size ${
                pX.shape[pX.shape.length - 1]
            }`
        );
    }
    if (reduceInfo.batchSize !== x.shape[0] * x.shape[1]) {
        throw new Error(
            `rmsNormGPU: batch size ${reduceInfo.batchSize} does not match expected size ${x.shape[0] * x.shape[1]}`
        );
    }

    const result: PackedTensorInfo = reduce(program, inputs, true, backend);
    result.packed = packed;

    if (packed && !packedX) {
        pX.dispose();
    }
    if (packed && !packedGamma) {
        pGamma.dispose();
    }

    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'RMSNorm',
    backendName: 'webgpu',
    kernelFunc: rmsNormGPU,
};

registerKernel(kernelConfig);
