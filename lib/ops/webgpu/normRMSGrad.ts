import {
    backend_util,
    engine,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { createReduceInfo } from './utils/reductions';
import { flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';
import { pack16 } from '../pack16';
import { isPackedTensor } from '@base/utilities/packed';
import { reshape16 } from '../reshape16';
import { sum16 } from '../sum16';
import { slice16 } from '../slice16';
import { unpack16 } from '../unpack16';
import { WebGPUProgram } from '@base/patches/webgpu_program';

class RMSGradProgram implements WebGPUProgram {
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
    packed = false;
    outputComponent?: number;

    constructor(reduceInfo: backend_util.ReduceInfo, rowsPerWorkgroup = 4, packed = false) {
        this.packed = packed;
        // this.outputComponent = packed ? 2 : 1;
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
          var<workgroup> accumulation: array<${this.packed ? 'vec2<f32>' : 'f32'}, 1024>;
        `;

        const sumRead = this.packed
            ? `
                let X = unpack2x16float(u32(x[offset + k]));
                let DY = unpack2x16float(u32(dy[offset + k]));
                let G  = unpack2x16float(u32(gamma[k]));
                sum_x2 = fma(X.x, X.x, sum_x2);
                sum_x2 = fma(X.y, X.y, sum_x2);
                sum_dygx = fma(DY.x * G.x, X.x, sum_dygx);
                sum_dygx = fma(DY.y * G.y, X.y, sum_dygx);
        `
            : `
                let X = f32(x[offset + k]);
                let DY = f32(dy[offset + k]);
                let G  = f32(gamma[k]);
                sum_x2 = fma(X, X, sum_x2);
                sum_dygx = fma(DY * G, X, sum_dygx);
            `;

        const writeDx = this.packed
            ? `
                let X  = unpack2x16float(u32(x[offset + k]));
                let DY = unpack2x16float(u32(dy[offset + k]));
                let G  = unpack2x16float(u32(gamma[k]));

                let dyGamma = DY * G;
                let dx = vec2<f32>(
                    fma(dyGamma.x, invRMS, -X.x * scale),
                    fma(dyGamma.y, invRMS, -X.y * scale)
                );

                result[offset + k] = i32(pack2x16float(dx));

                // dGamma
                accumulation[k] = fma(DY, X * invRMS, accumulation[k]);
        `
            : `
                let X  = f32(x[offset + k]);
                let DY = f32(dy[offset + k]);
                let G  = f32(gamma[k]);

                let dyGamma = DY * G;
                let dx = fma(dyGamma, invRMS, -X * scale);

                result[offset + k] = dx;

                // dGamma
                accumulation[k] = fma(DY, X * invRMS, accumulation[k]);
        `;

        const writeGamma = this.packed
            ? `
                result[outDgBase + k] = i32(pack2x16float(accumulation[k]));
        `
            : `
                result[outDgBase + k] = accumulation[k];
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
                accumulation[k] = ${this.packed ? 'vec2<f32>(0.0f)' : '0.0f'};
            }

            for (var rowOff = 0; rowOff < ${rowsPerWorkgroup}; rowOff = rowOff + 1) {
                let row = i32(workgroupId.x) * ${rowsPerWorkgroup} + rowOff;
                let offset = row * Length;
                
                var sum_x2 = 0.0f;
                var sum_dygx = 0.0f;

                for (var k = i32(localId.x); k < Length; k = k + ${workgroupSizeX}) {
                    ${sumRead}
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

                let invN = 1.0f / f32(${this.packed ? 'Length * 2' : 'Length'});
                let mean_x2   = fma(partials[0].x, invN, 1e-8);
                let mean_dygx = partials[0].y * invN;

                let invRMS = inverseSqrt(mean_x2);
                let scale = (mean_dygx / (mean_x2)) * invRMS;

                // write dx and dGamma.
                for (var k = i32(localId.x); k < Length; k = k + ${workgroupSizeX}) {
                    ${writeDx}
                }

                workgroupBarrier();
            }

            // Write out the partially accumulated dGamma
            let outDgBase = BatchSize * Length + i32(workgroupId.x) * Length;
            for (var k = i32(localId.x); k < Length; k = k + ${workgroupSizeX}) {
                ${writeGamma}
            }
          }
        `;
        return userCode;
    }
}

function rmsNormGradGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo[] {
    const { dy, x, gamma } = args.inputs as { dy: Tensor; x: Tensor; gamma: Tensor };

    const ROWS = 4;

    assertShapesMatch(x.shape, dy.shape, 'Error in RMSNormGrad dy: ');

    const packedX = isPackedTensor(x);
    const packedGamma = isPackedTensor(gamma);
    const packedDy = isPackedTensor(dy);
    const packed = packedX || packedGamma || packedDy;

    const pX = !packed || packedX ? x : pack16(x);
    const pGamma = !packed || packedGamma ? gamma : pack16(gamma);
    const pDy = !packed || packedDy ? dy : pack16(dy);

    assertShapesMatch(pGamma.shape, [pX.shape[pX.shape.length - 1]], 'Error in RMSNormGrad gamma: ');

    const backend = args.backend as WebGPUBackend;
    const reduceInfo = createReduceInfo([pX, pGamma, pDy], -1);

    const program = new RMSGradProgram(reduceInfo, ROWS, packed);
    const uniformData = [
        { type: 'int32', data: [program.inputShape[1]] }, // Reduce size
        { type: 'int32', data: [program.inputShape[0]] }, // Batch size
    ];

    if (reduceInfo.inSize > 1024) {
        throw new Error(`rmsNormGradGPU: inSize ${reduceInfo.inSize} exceeds max of 1024`);
    }

    const result = backend.runWebGPUProgram(program, [pX, pGamma, pDy], packed ? 'packedF16' : 'float32', uniformData);

    if (packed && !packedX) {
        pX.dispose();
    }
    if (packed && !packedGamma) {
        pGamma.dispose();
    }
    if (packed && !packedDy) {
        pDy.dispose();
    }

    const reduced = engine().makeTensorFromTensorInfo(result);

    // Now split reduced into dx and dGamma
    const dxDense = slice16(reduced, [0, 0], [reduceInfo.batchSize, reduceInfo.inSize]);
    const dGammaFullDense = slice16(
        reduced,
        [reduceInfo.batchSize, 0],
        [reduceInfo.batchSize / ROWS, reduceInfo.inSize]
    );
    reduced.dispose();

    const dx = reshape16(dxDense, x.shape);
    dxDense.dispose();

    const dGamma = sum16(dGammaFullDense, [0]);
    dGammaFullDense.dispose();

    return [dx, !packed || packedGamma ? dGamma : unpack16(dGamma)];
}

const gradKernelConfig: KernelConfig = {
    kernelName: 'RMSNormGrad',
    backendName: 'webgpu',
    kernelFunc: rmsNormGradGPU,
};

registerKernel(gradKernelConfig);
