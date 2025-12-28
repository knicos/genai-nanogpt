import { backend_util, engine, TensorInfo, util } from '@tensorflow/tfjs-core';
import { getMainHeaderString as main, WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { PackedTensorInfo } from '@base/patches/PackedTensor';
import { reshape16 } from '@base/ops/reshape16';

export interface ReduceWebGPUProgram extends WebGPUProgram {
    inputShape: number[];
    packed?: boolean;
    keepDims?: boolean;
}

function createReductionShader16_keepDims(
    workgroupSizeX: number,
    reductionOp: 'mean' | 'sum',
    inputSnippet: string,
    reducedSnippet: string,
    outputSnippet: string,
    inputReadSnippet?: string
): string {
    const sharedMemorySnippet = `
             var<workgroup> xBestValues : array<f32, ${workgroupSizeX}>;
           `;

    const userCode = `
           fn DIV_CEIL(a : u32, b : u32) -> u32 {
            return ((a - 1u) / b + 1u);
           }

            fn readInput(index: i32) -> vec2<f32> {
                ${
                    inputReadSnippet
                        ? inputReadSnippet
                        : `
                let packed = u32(x[index]);
                return unpack2x16float(packed);
                `
                }
            }

           ${sharedMemorySnippet}
    
           ${main('index')} {
                let outputIndex = index / ${workgroupSizeX};
                let offset = outputIndex * uniforms.reduceSize;
                var bestValue = 0.0f;
                let Length = uniforms.reduceSize;
    
                for (var k = i32(localId.x); k < Length;
                    k = k + ${workgroupSizeX}) {
                    var candidate = readInput(offset + k);
                    ${inputSnippet}
                    bestValue = bestValue + candidate.x + candidate.y;
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

                bestValue = xBestValues[0] ${reductionOp === 'mean' ? '/ f32(uniforms.reduceSize * 2i)' : ''};

                ${reducedSnippet}

                for (var k = i32(localId.x); k < Length;
                    k = k + ${workgroupSizeX}) {
                    ${outputSnippet}
                }
           }
         `;
    return userCode;
}

function createReductionShader16_flatten(
    workgroupSizeX: number,
    reductionOp: 'mean' | 'sum',
    inputSnippet: string,
    reducedSnippet: string,
    outputSnippet: string,
    inputReadSnippet?: string
): string {
    const sharedMemorySnippet = `
             var<workgroup> bestValues : array<vec2<f32>, ${workgroupSizeX}>;
           `;

    const userCode = `
           fn DIV_CEIL(a : u32, b : u32) -> u32 {
            return ((a - 1u) / b + 1u);
           }

           fn readInput(index: i32) -> vec2<f32> {
                ${
                    inputReadSnippet
                        ? inputReadSnippet
                        : `
                let packed = u32(x[index]);
                return unpack2x16float(packed);
                `
                }
            }

           ${sharedMemorySnippet}
    
           ${main('index')} {
                let outputIndex = index / ${workgroupSizeX};
                let offset1 = outputIndex * 2 * uniforms.reduceSize;
                let offset2 = offset1 + uniforms.reduceSize;
                var bestValue = vec2<f32>(0.0f, 0.0f);
                let Length = uniforms.reduceSize;
    
                for (var k = i32(localId.x); k < Length;
                    k = k + ${workgroupSizeX}) {
                    var candidate = readInput(offset1 + k);
                    ${inputSnippet}
                    let bv1 = candidate.x + candidate.y;

                    candidate = readInput(offset2 + k);
                    ${inputSnippet}
                    let bv2 = candidate.x + candidate.y;

                    bestValue = bestValue + vec2<f32>(bv1, bv2);
                }
                bestValues[localId.x] = bestValue;
                workgroupBarrier();
    
                var reduceSize = min(u32(Length), ${workgroupSizeX}u);
                for (var currentSize = reduceSize / 2u; reduceSize > 1u;
                    currentSize = reduceSize / 2u) {
                    let interval = DIV_CEIL(reduceSize, 2u);
                    if (localId.x < currentSize) {
                        let candidate = bestValues[localId.x + interval];
                        bestValue = bestValue + candidate;
                        bestValues[localId.x] = bestValue;
                    }
                    reduceSize = interval;
                    workgroupBarrier();
                }

                bestValue = bestValues[0] ${reductionOp === 'mean' ? '/ f32(uniforms.reduceSize * 2i)' : ''};

                ${reducedSnippet}
                ${outputSnippet}
           }
         `;
    return userCode;
}

export function createReductionShader16(
    workgroupSizeX: number,
    reductionOp: 'mean' | 'sum',
    inputSnippet: string,
    reducedSnippet: string,
    outputSnippet: string,
    inputReadSnippet?: string,
    keepDims = true
): string {
    return keepDims
        ? createReductionShader16_keepDims(
              workgroupSizeX,
              reductionOp,
              inputSnippet,
              reducedSnippet,
              outputSnippet,
              inputReadSnippet
          )
        : createReductionShader16_flatten(
              workgroupSizeX,
              reductionOp,
              inputSnippet,
              reducedSnippet,
              outputSnippet,
              inputReadSnippet
          );
}

export function createReductionShader32(
    workgroupSizeX: number,
    reductionOp: 'mean' | 'sum',
    inputSnippet: string,
    reducedSnippet: string,
    outputSnippet: string,
    inputReadSnippet?: string
): string {
    const sharedMemorySnippet = `
             var<workgroup> xBestValues : array<f32, ${workgroupSizeX}>;
           `;

    const userCode = `
           fn DIV_CEIL(a : u32, b : u32) -> u32 {
            return ((a - 1u) / b + 1u);
           }

           fn readInput(index: i32) -> f32 {
                ${
                    inputReadSnippet
                        ? inputReadSnippet
                        : `
                return x[index];
                `
                }
            }

           ${sharedMemorySnippet}
    
           ${main('index')} {
                let outputIndex = index / ${workgroupSizeX};
                let offset = outputIndex * uniforms.reduceSize;
                var bestValue = 0.0f;
                let Length = uniforms.reduceSize;
    
                for (var k = i32(localId.x); k < Length;
                    k = k + ${workgroupSizeX}) {
                    var candidate = readInput(offset + k);
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

export function reduce(
    program: ReduceWebGPUProgram,
    inputs: TensorInfo[],
    elementWise: boolean,
    backend: WebGPUBackend
): TensorInfo {
    const input = inputs[0];
    const inSize = program.inputShape[program.inputShape.length - 1];
    const uniformData = [{ type: 'int32', data: [inSize] }];

    const reduced: PackedTensorInfo = backend.runWebGPUProgram(
        program,
        inputs,
        program.packed ? 'int32' : 'float32',
        uniformData
    );
    reduced.packed = program.packed ?? false;
    const reducedTensor = engine().makeTensorFromTensorInfo(reduced);

    const res = reshape16(
        reducedTensor,
        elementWise
            ? input.shape
            : program.keepDims
            ? [...input.shape.slice(0, -2), input.shape[input.shape.length - 2] / 2, 1]
            : [...input.shape.slice(0, -2), input.shape[input.shape.length - 2] / 2]
    );
    reducedTensor.dispose();

    return res;
}
