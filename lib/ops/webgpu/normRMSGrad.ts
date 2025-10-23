import {
    backend_util,
    engine,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    slice,
    sum,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { createReduceInfo, ReduceWebGPUProgram } from './utils/reductions';
import { flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';

class RMSGradProgram implements ReduceWebGPUProgram {
    outputShape: number[];
    shaderKey = 'RMSNormGrad';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    variableNames = ['x', 'gamma', 'dy'];
    uniforms = 'reduceSize : i32, batchSize: i32';
    inputShape: number[];
    size = false;
    rowsPerWorkgroup: number;

    constructor(reduceInfo: backend_util.ReduceInfo, rowsPerWorkgroup = 4) {
        this.shaderKey = `RMSNormGrad_${rowsPerWorkgroup}`;
        this.rowsPerWorkgroup = rowsPerWorkgroup;
        this.inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
        this.outputShape = [reduceInfo.batchSize + reduceInfo.batchSize / this.rowsPerWorkgroup, reduceInfo.inSize];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = [reduceInfo.batchSize / this.rowsPerWorkgroup, 1, 1];

        if (reduceInfo.batchSize % this.rowsPerWorkgroup !== 0) {
            throw new Error(
                `RMSNormGradProgram: batch size ${reduceInfo.batchSize} must be ` +
                    `divisible by rowsPerWorkgroup ${this.rowsPerWorkgroup}`
            );
        }

        if (reduceInfo.inSize > 1024) {
            throw new Error(`RMSNormGradProgram: inSize ${reduceInfo.inSize} exceeds max of 1024`);
        }
    }

    getUserCode(): string {
        const workgroupSizeX = this.workgroupSize[0];
        const rowsPerWorkgroup = this.rowsPerWorkgroup;
        //const groupsX = Math.ceil(this.inputShape[0] / rowsPerWorkgroup);

        // NOTE: This is far from ideal, hard coded accumulation size.
        const sharedMemorySnippet = `
          var<workgroup> partials : array<vec2<f32>, ${workgroupSizeX}>;
          var<workgroup> accumulation: array<f32, 1024>;
        `;

        const userCode = `
          fn DIV_CEIL(a : u32, b : u32) -> u32 {
            return ((a - 1u) / b + 1u);
          }

          ${sharedMemorySnippet}

          ${main('index')} {
            // One workgroup per row (batch).
            let Length = uniforms.reduceSize;
            let BatchSize = uniforms.batchSize;
            for (var k = i32(localId.x); k < Length; k = k + ${workgroupSizeX}) {
                accumulation[k] = 0.0;
            }

            for (var rowOff = 0; rowOff < ${rowsPerWorkgroup}; rowOff = rowOff + 1) {
                let row = i32(workgroupId.x) * ${rowsPerWorkgroup} + rowOff;
                let offset = row * Length;
                
                var sum_x2 = 0.0;
                var sum_dygx = 0.0;

                for (var k = i32(localId.x); k < Length; k = k + ${workgroupSizeX}) {
                    let X = f32(x[offset + k]);
                    let DY = f32(dy[offset + k]);
                    let G  = f32(gamma[k]);
                    sum_x2 = fma(X, X, sum_x2);
                    sum_dygx = fma(DY * G, X, sum_dygx);
                }

                partials[localId.x] = vec2<f32>(sum_x2, sum_dygx);
                workgroupBarrier();

                var reduceSize = min(u32(Length), ${workgroupSizeX}u);
                for (var currentSize = reduceSize / 2u; reduceSize > 1u; currentSize = reduceSize / 2u) {
                    let interval = DIV_CEIL(reduceSize, 2u);
                    if (localId.x < currentSize) {
                        partials[localId.x] = partials[localId.x] + partials[localId.x + interval];
                    }
                    reduceSize = interval;
                    workgroupBarrier();
                }

                let invN = 1.0 / f32(Length);
                let mean_x2   = partials[0].x * invN;
                let mean_dygx = partials[0].y * invN;

                let invRMS = inverseSqrt(mean_x2 + 1e-8);
                let scale = (mean_dygx / mean_x2) * invRMS;

                // write dx and dGamma.
                for (var k = i32(localId.x); k < Length; k = k + ${workgroupSizeX}) {
                    let X  = f32(x[offset + k]);
                    let DY = f32(dy[offset + k]);
                    let G  = f32(gamma[k]);

                    let dyGamma = DY * G;
                    let dx = fma(dyGamma, invRMS, -X * scale);

                    result[offset + k] = dx;

                    // dGamma
                    accumulation[k] = fma(DY, X * invRMS, accumulation[k]);
                }

                workgroupBarrier();
            }

            // Write out the partially accumulated dGamma
            let outDgBase = BatchSize * Length + i32(workgroupId.x) * Length;
            for (var k = i32(localId.x); k < Length; k = k + ${workgroupSizeX}) {
                result[outDgBase + k] = accumulation[k];
            }
          }
        `;
        return userCode;
    }
}

function rmsNormGradGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo[] {
    const { dy, x, gamma } = args.inputs as { dy: Tensor; x: Tensor; gamma: Tensor };

    const ROWS = 4;

    const backend = args.backend as WebGPUBackend;
    const reduceInfo = createReduceInfo([x, gamma, dy], -1);
    const program = new RMSGradProgram(reduceInfo, ROWS);
    const uniformData = [
        { type: 'int32', data: [program.inputShape[1]] }, // Reduce size
        { type: 'int32', data: [program.inputShape[0]] }, // Batch size
    ];

    const reduced = engine().makeTensorFromTensorInfo(
        backend.runWebGPUProgram(program, [x, gamma, dy], 'float32', uniformData)
    );

    // Now split reduced into dx and dGamma
    const dxDense = slice(reduced, [0, 0], [reduceInfo.batchSize, reduceInfo.inSize]);
    const dGammaFullDense = slice(reduced, [reduceInfo.batchSize, 0], [reduceInfo.batchSize / ROWS, reduceInfo.inSize]);
    reduced.dispose();

    const dx = dxDense.reshape(x.shape);
    dxDense.dispose();

    const dGamma = sum(dGammaFullDense, [0]);
    dGammaFullDense.dispose();

    return [dx, dGamma];
}

const gradKernelConfig: KernelConfig = {
    kernelName: 'RMSNormGrad',
    backendName: 'webgpu',
    kernelFunc: rmsNormGradGPU,
};

registerKernel(gradKernelConfig);
