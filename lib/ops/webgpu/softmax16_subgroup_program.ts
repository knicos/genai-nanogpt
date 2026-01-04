import { WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';

export default class SoftmaxSubgroupProgram implements WebGPUProgram {
    variableNames = ['logits'];
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number];
    minSubgroupSize: number;
    maxSubgroupSize: number;
    subgroups = true;
    subgroupBuiltins = false;

    constructor(outputShape: number[], minSubgroupSize: number, maxSubgroupSize: number) {
        this.minSubgroupSize = minSubgroupSize;
        this.maxSubgroupSize = maxSubgroupSize;
        this.outputShape = outputShape; // [rows, cols]
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = [this.outputShape[0], 1, 1];
        this.workgroupSize = [Math.min(64, maxSubgroupSize), 1, 1];

        if (minSubgroupSize !== maxSubgroupSize) {
            this.subgroupBuiltins = true;
            this.workgroupSize = [64, 1, 1];
        }

        this.shaderKey = 'softmax16subgroup';
    }

    getUserCode(): string {
        const useSubgroupSize = this.maxSubgroupSize !== this.minSubgroupSize;

        const userCode = `
        ${useSubgroupSize ? `var<workgroup> bestValues : array<f32, ${this.workgroupSize[0]}>;` : ''}
        const blockSize = ${this.workgroupSize[0]};
        ${main('index')} {
            let row = index / blockSize;
            let tid = i32(localId.x);
            let cols = uniforms.outShape[1];
            let rowIdx = row * cols;

            var threadMax = -3.402823e+38f;
            for (var col = tid; col < cols; col += blockSize) {
                let value = unpack2x16float(u32(logits[rowIdx + col]));
                threadMax = max(threadMax, max(value.x, value.y));
            }

            threadMax = subgroupMax(threadMax);
            ${useSubgroupSize ? `
                let lane = localId.x % subgroupSize;
                if (lane == 0) {
                    bestValues[localId.x / subgroupSize] = threadMax;
                }
                workgroupBarrier();
                let numSubgroups = blockSize / subgroupSize;
                threadMax = select(-3.402823e+38f, bestValues[lane], lane < numSubgroups);
                threadMax = subgroupMax(threadMax);
                workgroupBarrier();    
            `: ''}

            var threadSum = 0.0f;
            for (var col = tid; col < cols; col += blockSize) {
                let value = unpack2x16float(u32(logits[rowIdx + col]));
                let subExp = exp(value - threadMax);
                threadSum += subExp.x + subExp.y;
            }

            threadSum = subgroupAdd(threadSum);
            ${useSubgroupSize ? `
                if (lane == 0) {
                    bestValues[localId.x / subgroupSize] = threadSum;
                }
                workgroupBarrier();
                threadSum = select(0.0f, bestValues[lane], lane < numSubgroups);
                threadSum = subgroupAdd(threadSum);    
            `: ''}

            for (var col = tid; col < cols; col += ${useSubgroupSize ? 'i32(subgroupSize)' : 'blockSize'}) {
                let value = unpack2x16float(u32(logits[rowIdx + col]));
                let valuePair: vec2<f32> = exp(value - threadMax) / threadSum;
                result[rowIdx + col] = i32(pack2x16float(valuePair));
            }
        }
      `;
        return userCode;
    }
}
