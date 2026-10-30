import RoPECache from '@base/layers/RoPECache';
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
    interleaved: boolean;

    constructor(batch: number, heads: number, T: number, C: number, interleaved = true) {
        this.shaderKey = `Rope_${C}_${interleaved ? 'interleaved' : 'split'}`;
        this.outputShape = [batch, heads, T, C];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.interleaved = interleaved;
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
                    let half = rotaryDim / 2;
                    let idx = (t + uniforms.pastLen + b) * uniforms.cosShape[1] +
                        ${this.interleaved ? 'd / 2' : 'd % half'};
                    let cos = cos[idx];
                    let sin = sin[idx];

                    let ownX = x[xIdx] * cos;
                    var evenOdd = 0.0;

                    if (${this.interleaved ? 'd % 2 == 0' : 'd < half'}) {
                        evenOdd = -x[${this.interleaved ? 'xIdx + 1' : 'xIdx + half'}];
                    } else {
                        evenOdd = x[${this.interleaved ? 'xIdx - 1' : 'xIdx - half'}];
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

class RopeProgram16Interleaved implements WebGPUProgram {
    variableNames = ['x', 'sin', 'cos'];
    outputShape: number[];
    shaderKey = 'Rope';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    uniforms = 'pastLen: i32';

    constructor(batch: number, heads: number, T: number, C: number) {
        this.shaderKey = `Rope_${C}_interleaved_packed`;
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

                let xPair = unpack2x16float(u32(x[xIdx]));
                let idx = (t + uniforms.pastLen + b) * uniforms.cosShape[1] + d;
                let cosVal = cos[idx];
                let sinVal = sin[idx];

                outVal = vec2<f32>(
                    fma(-xPair.y, sinVal, xPair.x * cosVal),
                    fma(xPair.x, sinVal, xPair.y * cosVal)
                );

                result[index] = i32(pack2x16float(outVal));
            }
        }
        `;
    }
}

class RopeProgram16Split implements WebGPUProgram {
    variableNames = ['x', 'sin', 'cos'];
    outputShape: number[];
    shaderKey = 'Rope';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workPerThread = 2;
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    uniforms = 'pastLen: i32';

    constructor(batch: number, heads: number, T: number, C: number) {
        this.shaderKey = `Rope_${C}_split_packed`;
        // C / 2 due to packing
        this.outputShape = [batch, heads, T, C / 2];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [
            this.workPerThread,
            1,
            1,
        ]);
    }

    getUserCode() {
        return `
        ${main('index')} {
            let halfPacked = uniforms.outShape[3] / 2;
            for (var i = 0; i < ${this.workPerThread}; i = i + 1) {
                let flatIndex = index * ${this.workPerThread} + i;
                if (flatIndex < uniforms.size) {
                    let coords = getCoordsFromIndex(flatIndex); // [b, h, t, dPacked]
                    let d = coords[3];

                    if (d < halfPacked) {
                        let t = coords[2];
                        let xLow = unpack2x16float(u32(x[flatIndex]));
                        let xHigh = unpack2x16float(u32(x[flatIndex + halfPacked]));

                        let scalar0 = d * 2;
                        let scalar1 = scalar0 + 1;
                        let cacheOffset = (t + uniforms.pastLen + coords[0]) * uniforms.cosShape[1];

                        let cos0 = cos[cacheOffset + scalar0];
                        let sin0 = sin[cacheOffset + scalar0];
                        let cos1 = cos[cacheOffset + scalar1];
                        let sin1 = sin[cacheOffset + scalar1];

                        let outLow = vec2<f32>(
                            fma(-xHigh.x, sin0, xLow.x * cos0),
                            fma(-xHigh.y, sin1, xLow.y * cos1)
                        );
                        let outHigh = vec2<f32>(
                            fma(xLow.x, sin0, xHigh.x * cos0),
                            fma(xLow.y, sin1, xHigh.y * cos1)
                        );

                        result[flatIndex] = i32(pack2x16float(outLow));
                        result[flatIndex + halfPacked] = i32(pack2x16float(outHigh));
                    }
                }
            }
        }
        `;
    }
}

function ropeGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { pastLen, negSin, ropeCache, interleaved } = args.attrs as unknown as {
        pastLen: number;
        negSin: boolean;
        ropeCache: RoPECache;
        interleaved: boolean;
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
        ? interleaved
            ? new RopeProgram16Interleaved(batchSize, heads, seqLength, C)
            : new RopeProgram16Split(batchSize, heads, seqLength, C)
        : new RopeProgram32(batchSize, heads, seqLength, C, interleaved);

    const uniformData = [{ type: 'int32', data: [pastLen] }];
    const dtype = packed ? 'packedF16' : x.dtype;
    const result = backend.runWebGPUProgram(program, [x, sin, cos], dtype, uniformData);
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'Rope',
    backendName: 'webgpu',
    kernelFunc: ropeGPU,
};

registerKernel(kernelConfig);
