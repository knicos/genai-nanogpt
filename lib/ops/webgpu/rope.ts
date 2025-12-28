import RoPECache from '@base/layers/RoPECache';
import { PackedTensorInfo } from '@base/patches/PackedTensor';
import { isPackedTensor } from '@base/utilities/packed';
import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';

import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';

class RopeProgram32 implements WebGPUProgram {
    variableNames = ['x', 'sin', 'cos'];
    outputShape: number[];
    shaderKey = 'Rope';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    uniforms = 'pastLen: i32';

    constructor(batch: number, heads: number, T: number, C: number) {
        this.shaderKey = `Rope_${C}`;
        this.outputShape = [batch, heads, T, C];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode() {
        const C = this.outputShape[3];
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index); // [b, h, t, d]
                let b = coords[0];
                let h = coords[1];
                let t = coords[2];
                let d = coords[3];

                let rotaryDim = ${C};

                var outVal = 0.0;

                let xIdx = b * uniforms.outShapeStrides[0] +
                    h * uniforms.outShapeStrides[1] +
                    t * uniforms.outShapeStrides[2] +
                    d;

                if (d < rotaryDim) {
                    let idx = (t + uniforms.pastLen) * uniforms.cosShape[1] + d / 2;
                    let cos = cos[idx];
                    let sin = sin[idx];

                    let ownX = x[xIdx] * cos;
                    var evenOdd = 0.0;

                    if (d % 2 == 0) {
                        // even index
                        evenOdd = -x[xIdx + 1];
                    } else {
                        // odd index
                        evenOdd = x[xIdx - 1];
                    }

                    outVal = fma(evenOdd, sin, ownX);
                } else {
                    // pass through for non-rotary dims
                    outVal = x[xIdx];
                }

                setOutputAtIndex(index, outVal);
            }
        }
        `;
    }
}

class RopeProgram16 implements WebGPUProgram {
    variableNames = ['x', 'sin', 'cos'];
    outputShape: number[];
    shaderKey = 'Rope';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    uniforms = 'pastLen: i32';

    constructor(batch: number, heads: number, T: number, C: number) {
        this.shaderKey = `Rope_${C}`;
        // C / 2 due to packing
        this.outputShape = [batch, heads, T, C / 2];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode() {
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index); // [b, h, t, d]
                let b = coords[0];
                let h = coords[1];
                let t = coords[2];
                let d = coords[3];

                var outVal = vec2<f32>(0.0, 0.0);

                let xIdx = b * uniforms.outShapeStrides[0] +
                    h * uniforms.outShapeStrides[1] +
                    t * uniforms.outShapeStrides[2] +
                    d;

                let idx = (t + uniforms.pastLen) * uniforms.cosShape[1] + d;
                let cos = cos[idx];
                let sin = sin[idx];

                let xPair = unpack2x16float(u32(x[xIdx]));
                let ownX = vec2<f32>(xPair.x * cos, xPair.y * cos);

                let evenOdd = vec2<f32>(
                    -xPair.y,
                    xPair.x
                );

                outVal = vec2<f32>(
                    fma(evenOdd.x, sin, ownX.x),
                    fma(evenOdd.y, sin, ownX.y)
                );

                result[index] = i32(pack2x16float(outVal));
            }
        }
        `;
    }
}

function ropeGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { pastLen, negSin, ropeCache } = args.attrs as unknown as {
        pastLen: number;
        negSin: boolean;
        ropeCache: RoPECache;
    };

    const backend = args.backend as WebGPUBackend;

    const packed = isPackedTensor(x);
    const batchSize = x.shape[0];
    const heads = x.shape[1]!;
    const seqLength = x.shape[2]!;
    // C is doubled due to packing
    const C = packed ? x.shape[3]! * 2 : x.shape[3]!;

    const sin = negSin ? ropeCache.getNegSin()! : ropeCache.getSin()!;
    const cos = ropeCache.getCos()!;

    assertShapesMatch(sin.shape, cos.shape, 'Error in Rope: ');
    if (sin.shape[0] < seqLength + pastLen) {
        throw new Error(
            `Sin tensor shape ${sin.shape} is not compatible with seqLength ${seqLength} and pastLen ${pastLen}.`
        );
    }
    if (sin.shape[1]! * 2 < C) {
        throw new Error(`Sin tensor shape ${sin.shape} is not compatible with feature dimension ${C}.`);
    }
    if (sin.shape.length !== 3) {
        throw new Error(`Sin tensor must be 3-dimensional, but got shape ${sin.shape}.`);
    }

    const program = packed
        ? new RopeProgram16(batchSize, heads, seqLength, C)
        : new RopeProgram32(batchSize, heads, seqLength, C);

    const uniformData = [{ type: 'int32', data: [pastLen] }];
    const dtype = packed ? 'int32' : x.dtype;
    const result: PackedTensorInfo = backend.runWebGPUProgram(program, [x, sin, cos], dtype, uniformData);
    result.packed = packed;
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'Rope',
    backendName: 'webgpu',
    kernelFunc: ropeGPU,
};

registerKernel(kernelConfig);
