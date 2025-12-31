import { backend_util, engine, TensorInfo, util } from '@tensorflow/tfjs-core';
import { getMainHeaderString as main, WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { PackedTensorInfo } from '@base/patches/PackedTensor';
import { reshape16 } from '@base/ops/reshape16';
import { DeviceInformation } from './deviceInfo';
import { flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';

export interface ReduceParams {
    reductionOp: 'mean' | 'sum';
    elementwise?: boolean;
}

interface ReduceShaderParams extends ReduceParams {
    workgroupSizeX?: number;
    subgroups: boolean;
    variableSubgroups: boolean;
    inputSnippet: string;
    reducedSnippet?: string;
    outputSnippet: string;
    inputReadSnippet?: string;
}

function createReduceSnippet(subgroups: boolean, workgroupSizeX: number): string {
    if (subgroups) {
        return `
            bestValue = subgroupAdd(bestValue);
        `;
    } else {
        return `
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

            bestValue = bestValues[0];
        `;
    }
}

function createReductionShader16_elementwise(params: ReduceShaderParams): string {
    const reduceSize = params.variableSubgroups ? 'i32(subgroupSize)' : `${params.workgroupSizeX}`;
    const sharedMemorySnippet = params.subgroups
        ? ''
        : `
             var<workgroup> bestValues : array<f32, ${params.workgroupSizeX}>;
           `;

    const reduction = createReduceSnippet(params.subgroups, params.workgroupSizeX!);

    const userCode = `
           fn DIV_CEIL(a : u32, b : u32) -> u32 {
            return ((a - 1u) / b + 1u);
           }

            fn readInput(index: i32) -> vec2<f32> {
                ${
                    params.inputReadSnippet
                        ? params.inputReadSnippet
                        : `
                let packed = u32(x[index]);
                return unpack2x16float(packed);
                `
                }
            }

           ${sharedMemorySnippet}
    
           ${main('index')} {
                let outputIndex = index / ${reduceSize};
                let offset = outputIndex * uniforms.reduceSize;
                var bestValue = 0.0f;
                let Length = uniforms.reduceSize;
                let tid = i32(${params.variableSubgroups ? 'subgroupInvocationId' : 'localId.x'});
    
                for (var k = tid; k < Length;
                    k = k + ${reduceSize}) {
                    var candidate = readInput(offset + k);
                    ${params.inputSnippet}
                    bestValue = bestValue + candidate.x + candidate.y;
                }

                ${reduction}
                bestValue = bestValue ${params.reductionOp === 'mean' ? '/ f32(uniforms.reduceSize * 2i)' : ''};

                ${params.reducedSnippet ? params.reducedSnippet : ''}

                for (var k = tid; k < Length;
                    k = k + ${reduceSize}) {
                    ${params.outputSnippet}
                }
           }
         `;
    return userCode;
}

function createReductionShader16_flatten(params: ReduceShaderParams): string {
    const reduceSize = params.variableSubgroups ? 'i32(subgroupSize)' : `${params.workgroupSizeX}`;
    const sharedMemorySnippet = params.subgroups
        ? ''
        : `
             var<workgroup> bestValues : array<vec2<f32>, ${params.workgroupSizeX}>;
           `;

    const reduction = createReduceSnippet(params.subgroups, params.workgroupSizeX!);

    const userCode = `
           fn DIV_CEIL(a : u32, b : u32) -> u32 {
            return ((a - 1u) / b + 1u);
           }

           fn readInput(index: i32) -> vec2<f32> {
                ${
                    params.inputReadSnippet
                        ? params.inputReadSnippet
                        : `
                let packed = u32(x[index]);
                return unpack2x16float(packed);
                `
                }
            }

           ${sharedMemorySnippet}
    
           ${main('index')} {
                let outputIndex = index / ${reduceSize};
                let offset1 = outputIndex * 2 * uniforms.reduceSize;
                let offset2 = offset1 + uniforms.reduceSize;
                var bestValue = vec2<f32>(0.0f, 0.0f);
                let Length = uniforms.reduceSize;
                let tid = i32(${params.variableSubgroups ? 'subgroupInvocationId' : 'localId.x'});
    
                for (var k = tid; k < Length;
                    k = k + ${reduceSize}) {
                    var candidate = readInput(offset1 + k);
                    ${params.inputSnippet}
                    let bv1 = candidate.x + candidate.y;

                    candidate = readInput(offset2 + k);
                    ${params.inputSnippet}
                    let bv2 = candidate.x + candidate.y;

                    bestValue = bestValue + vec2<f32>(bv1, bv2);
                }
                ${reduction}
                bestValue = bestValue ${params.reductionOp === 'mean' ? '/ f32(uniforms.reduceSize * 2i)' : ''};

                ${params.reducedSnippet ?? ''}
                ${params.outputSnippet}
           }
         `;
    return userCode;
}

function createReductionShader16(params: ReduceShaderParams): string {
    return params.elementwise ? createReductionShader16_elementwise(params) : createReductionShader16_flatten(params);
}

function createReductionShader32(params: ReduceShaderParams): string {
    const reduceSize = params.variableSubgroups ? 'i32(subgroupSize)' : `${params.workgroupSizeX}`;
    const sharedMemorySnippet = `
             var<workgroup> bestValues : array<f32, ${params.workgroupSizeX}>;
           `;

    const reduction = createReduceSnippet(params.subgroups, params.workgroupSizeX!);

    const userCode = `
           fn DIV_CEIL(a : u32, b : u32) -> u32 {
            return ((a - 1u) / b + 1u);
           }

           fn readInput(index: i32) -> f32 {
                ${
                    params.inputReadSnippet
                        ? params.inputReadSnippet
                        : `
                return x[index];
                `
                }
            }

           ${sharedMemorySnippet}
    
           ${main('index')} {
                let outputIndex = index / ${reduceSize};
                let offset = outputIndex * uniforms.reduceSize;
                var bestValue = 0.0f;
                let Length = uniforms.reduceSize;
                let tid = i32(${params.variableSubgroups ? 'subgroupInvocationId' : 'localId.x'});
    
                for (var k = tid; k < Length;
                    k = k + ${reduceSize}) {
                    var candidate = readInput(offset + k);
                    ${params.inputSnippet}
                    bestValue = bestValue + candidate;
                }
                ${reduction}

                bestValue = bestValue ${params.reductionOp === 'mean' ? '/ f32(uniforms.reduceSize)' : ''};

                ${params.reducedSnippet}

                for (var k = tid; k < Length;
                    k = k + ${reduceSize}) {
                    ${params.outputSnippet}
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

export class ReduceProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey = 'reduce16';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    variableNames = ['x'];
    uniforms = 'reduceSize : i32,';
    inputShape: number[];
    size = false;
    packed = true;
    outputComponent: number;
    variableComponents?: number[];
    elementwise: boolean;
    subgroups = false;
    subgroupBuiltins = false;
    deviceInfo: DeviceInformation;
    params: ReduceParams;

    constructor(
        deviceInfo: DeviceInformation,
        reduceInfo: backend_util.ReduceInfo,
        params: ReduceParams,
        packed: boolean
    ) {
        this.params = params;
        this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
        this.deviceInfo = deviceInfo;
        this.packed = packed;
        if (deviceInfo.subgroupsSupported) {
            this.workgroupSize = [Math.min(64, deviceInfo.subgroupMaxSize), 1, 1];
            this.subgroups = true;
            if (deviceInfo.variableSubgroups) {
                this.subgroupBuiltins = true;
            }
        }
        this.outputShape = params.elementwise ? [reduceInfo.batchSize, reduceInfo.inSize] : [reduceInfo.batchSize / 2];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = [params.elementwise ? reduceInfo.batchSize : reduceInfo.batchSize / 2, 1, 1];
        this.outputComponent = 1;
        this.variableComponents = [1];
        this.elementwise = params.elementwise === true;
    }

    protected getWriteSnippet(): string {
        return this.packed
            ? `result[outputIndex] = i32(pack2x16float(bestValue));`
            : `result[outputIndex] = bestValue;`;
    }

    protected getPreprocessSnippet(): string {
        return '';
    }

    protected getPostprocessSnippet(): string {
        return '';
    }

    protected getReadSnippet(): string {
        return this.packed
            ? `
                let packed = u32(x[index]);
                return unpack2x16float(packed);
                `
            : `return x[index];`;
    }

    getUserCode(): string {
        const workgroupSizeX = this.workgroupSize[0];

        const shader = this.packed
            ? createReductionShader16({
                  ...this.params,
                  workgroupSizeX,
                  subgroups: this.subgroups,
                  variableSubgroups: this.deviceInfo.variableSubgroups,
                  inputReadSnippet: this.getReadSnippet(),
                  inputSnippet: this.getPreprocessSnippet(),
                  outputSnippet: this.getWriteSnippet(),
                  reducedSnippet: this.getPostprocessSnippet(),
              })
            : createReductionShader32({
                  ...this.params,
                  workgroupSizeX,
                  subgroups: this.subgroups,
                  variableSubgroups: this.deviceInfo.variableSubgroups,
                  inputReadSnippet: this.getReadSnippet(),
                  inputSnippet: this.getPreprocessSnippet(),
                  outputSnippet: this.getWriteSnippet(),
                  reducedSnippet: this.getPostprocessSnippet(),
              });

        return shader;
    }
}

export function reduce(program: ReduceProgram, inputs: TensorInfo[], backend: WebGPUBackend): TensorInfo {
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
        program.elementwise
            ? input.shape
            : program.packed
            ? [...input.shape.slice(0, -2), input.shape[input.shape.length - 2] / 2]
            : [...input.shape.slice(0, -2), input.shape[input.shape.length - 2]]
    );
    reducedTensor.dispose();

    return res;
}
