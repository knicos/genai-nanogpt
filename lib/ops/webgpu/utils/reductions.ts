import { backend_util, TensorInfo, util } from '@tensorflow/tfjs-core';
import { getMainHeaderString as main, WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { reshape } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/Reshape';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

export interface ReduceWebGPUProgram extends WebGPUProgram {
    inputShape: number[];
}

export function createReductionShader(
    workgroupSizeX: number,
    reductionOp: 'mean' | 'sum',
    inputSnippet: string,
    reducedSnippet: string,
    outputSnippet: string
): string {
    const sharedMemorySnippet = `
             var<workgroup> xBestValues : array<f32, ${workgroupSizeX}>;
           `;

    const userCode = `
           fn DIV_CEIL(a : u32, b : u32) -> u32 {
            return ((a - 1u) / b + 1u);
           }

           ${sharedMemorySnippet}
    
           ${main('index')} {
                let outputIndex = index / ${workgroupSizeX};
                let offset = outputIndex * uniforms.reduceSize;
                var bestValue = 0.0;
                let Length = uniforms.reduceSize;
    
                for (var k = i32(localId.x); k < Length;
                    k = k + ${workgroupSizeX}) {
                    var candidate = f32(x[offset + k]);
                    ${inputSnippet}
                    bestValue = bestValue + candidate;
                }
                xBestValues[localId.x] = bestValue;
                workgroupBarrier();
    
                var reduceSize = min(u32(Length), ${workgroupSizeX}u);
                for (var currentSize = reduceSize / 2u; reduceSize > 1u;
                    currentSize = reduceSize / 2u) {
                    let interval = DIV_CEIL(reduceSize, 2u);
                    if (localId.x < currentSize) {
                        let candidate = xBestValues[localId.x + interval];
                        bestValue = bestValue + candidate;
                        xBestValues[localId.x] = bestValue;
                    }
                    reduceSize = interval;
                    workgroupBarrier();
                }

                bestValue = xBestValues[0] ${reductionOp === 'mean' ? '/ f32(uniforms.reduceSize)' : ''};

                ${reducedSnippet}

                for (var k = i32(localId.x); k < Length;
                    k = k + ${workgroupSizeX}) {
                    ${outputSnippet}
                }
           }
         `;
    return userCode;
}

export function createReduceInfo(inputs: TensorInfo[], axis: number | number[]): backend_util.ReduceInfo {
    const input = inputs[0];
    const origAxes = util.parseAxisParam(axis, input.shape);
    const axes = origAxes;

    const [, reduceShape] = backend_util.computeOutAndReduceShapes(input.shape, axes);
    const inSize = util.sizeFromShape(reduceShape);
    const xSize = util.sizeFromShape(input.shape);
    const batchSize = xSize / inSize;

    const reduceInfo = { windowSize: inSize, inSize, batchSize, outSize: 1 };
    return reduceInfo;
}

export function reduce(program: ReduceWebGPUProgram, inputs: TensorInfo[], backend: WebGPUBackend): TensorInfo {
    const toDispose: TensorInfo[] = [];

    const input = inputs[0];
    const inSize = program.inputShape[1];
    const uniformData = [{ type: 'int32', data: [inSize] }];

    const reduced = backend.runWebGPUProgram(program, inputs, 'float32', uniformData);
    toDispose.push(reduced);

    const res = reshape({ inputs: { x: reduced }, attrs: { shape: input.shape }, backend });

    toDispose.forEach((t) => backend.disposeData(t.dataId));

    return res;
}
