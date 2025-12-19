import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import {
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';

class QKVProgram16 implements WebGPUProgram {
    variableNames = ['x', 'kernel'];
    outputShape: number[];
    shaderKey = 'QKV';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    uniforms = 'mode: i32';
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;

    constructor(batch: number, nh: number, T: number, C: number) {
        // Output shape: [batch, nh, T, T]
        const head_dim = C / nh;
        this.shaderKey = `QKV_${nh}_${head_dim}`;
        this.outputShape = [batch, nh, T, head_dim / 2];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode() {
        const nh = this.outputShape[1];
        const head_dim = this.outputShape[3] * 2;
        const C = nh * head_dim;
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index); // [b, h, t, d]
                let b = coords[0];
                let h = coords[1];
                let t = coords[2];
                let d = coords[3];

                // Compute output channel index in fused kernel
                let out_offset = uniforms.mode * ${C} + h * ${head_dim} + d * 2;

                var sum = vec2(0.0, 0.0);
            
                let baseX = b * uniforms.xShape[1] * uniforms.xShape[2] + t * uniforms.xShape[2];
                for (var c = 0; c < ${C}; c += 1) {
                    let xval = x[baseX + c];
                    let kval1 = getKernel(c, out_offset);
                    let kval2 = getKernel(c, out_offset + 1);
                    sum[0] = fma(xval, kval1, sum[0]);
                    sum[1] = fma(xval, kval2, sum[1]);
                }
                result[index] = i32(pack2x16float(sum));
            }
        }
        `;
    }
}

class QKVProgram32 implements WebGPUProgram {
    variableNames = ['x', 'kernel'];
    outputShape: number[];
    shaderKey = 'QKV';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    uniforms = 'mode: i32';
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;

    constructor(batch: number, nh: number, T: number, C: number) {
        // Output shape: [batch, nh, T, T]
        const head_dim = C / nh;
        this.shaderKey = `QKV_${nh}_${head_dim}`;
        this.outputShape = [batch, nh, T, head_dim];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode() {
        const nh = this.outputShape[1];
        const head_dim = this.outputShape[3];
        const C = nh * head_dim;
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index); // [b, h, t, d]
                let b = coords[0];
                let h = coords[1];
                let t = coords[2];
                let d = coords[3];

                // Compute output channel index in fused kernel
                let out_offset = uniforms.mode * ${C} + h * ${head_dim} + d;

                var sum = 0.0;
                let baseX = b * uniforms.xShape[1] * uniforms.xShape[2] + t * uniforms.xShape[2];
                for (var c = 0; c < ${C}; c += 1) {
                    let xval = x[baseX + c];
                    let kval = getKernel(c, out_offset);
                    sum = fma(xval, kval, sum);
                }

                setOutputAtIndex(index, sum);
            }
        }
        `;
    }
}

function qkvGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo[] {
    const { x, kernel } = args.inputs as { x: Tensor; kernel: Tensor };
    const { heads, packed } = args.attrs as { heads: number; packed?: boolean };

    const backend = args.backend as WebGPUBackend;

    const batchSize = x.shape[0];
    const seqLength = x.shape[1]!;
    const C = x.shape[2]!;

    assertShapesMatch(kernel.shape, [C, 3 * C], 'Error in QKV: ');
    if (C % heads !== 0) {
        throw new Error(`Channel dimension ${C} must be divisible by number of heads ${heads} in QKV.`);
    }

    const program = packed
        ? new QKVProgram16(batchSize, heads, seqLength, C)
        : new QKVProgram32(batchSize, heads, seqLength, C);
    const dtype = packed ? 'int32' : x.dtype;
    return [
        backend.runWebGPUProgram(program, [x, kernel], dtype, [{ type: 'int32', data: [0] }]),
        backend.runWebGPUProgram(program, [x, kernel], dtype, [{ type: 'int32', data: [1] }]),
        backend.runWebGPUProgram(program, [x, kernel], dtype, [{ type: 'int32', data: [2] }]),
    ];
}

const kernelConfig: KernelConfig = {
    kernelName: 'QKV',
    backendName: 'webgpu',
    kernelFunc: qkvGPU,
};

registerKernel(kernelConfig);
