import { flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { backend_util } from '@tensorflow/tfjs-core';
import { createReductionShader16, ReduceWebGPUProgram } from './utils/reductions';

export default class RMSProgram16 implements ReduceWebGPUProgram {
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
