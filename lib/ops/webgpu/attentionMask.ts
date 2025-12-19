import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';
import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';
import { isPackedTensor } from '@base/utilities/packed';
import { PackedTensorInfo } from '@base/patches/PackedTensor';

class AttentionMaskProgram32 implements WebGPUProgram {
    variableNames = ['q', 'k'];
    outputShape: number[];

    shaderKey = 'AttentionMask';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    uniforms = 'divisor: f32, pastLen: i32, inf: f32';
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    hs: number;
    nh: number;
    T1: number;
    T2: number;

    constructor(batch: number, nh: number, T1: number, T2: number, hs: number) {
        this.shaderKey = `AttentionMask_${hs}`;
        this.outputShape = [batch, nh, T1, T2];
        this.hs = hs;
        this.nh = nh;
        this.T1 = T1;
        this.T2 = T2;

        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);

        if (hs % 4 !== 0) {
            throw new Error('Head size must be a multiple of 4 for AttentionMaskProgram');
        }
    }

    getUserCode(): string {
        const userCode = `
            ${main('index')} {
                
                let coords = getCoordsFromIndex(index);
                let b = coords[0];
                let h = coords[1];
                let t1 = coords[2];
                let t2 = coords[3];

                if (index < uniforms.size) {
                    if (t2 > t1 + uniforms.pastLen) {
                        setOutputAtIndex(index, uniforms.inf);
                        return;
                    }

                    let q0 = getIndexFromCoords4D(vec4<i32>(b, h, t1, 0), uniforms.qShape);
                    let k0 = getIndexFromCoords4D(vec4<i32>(b, h, t2, 0), uniforms.kShape);
                    
                    var sum: f32 = 0.0;
                    for (var i: i32 = 0; i < ${this.hs}; i = i + 4) {
                        let qv = vec4<f32>(q[q0 + i], q[q0 + i + 1], q[q0 + i + 2], q[q0 + i + 3]);
                        let kv = vec4<f32>(k[k0 + i], k[k0 + i + 1], k[k0 + i + 2], k[k0 + i + 3]);
                        sum = sum + dot(qv, kv);
                    }
                    let scaled = sum * uniforms.divisor;
                    setOutputAtIndex(index, scaled);
                }
            }
        `;
        return userCode;
    }
}

class AttentionMaskProgram16 implements WebGPUProgram {
    variableNames = ['q', 'k'];
    outputShape: number[];

    shaderKey = 'AttentionMask';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    uniforms = 'divisor: f32, pastLen: i32, inf: f32';
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    hs: number;
    nh: number;
    T1: number;
    T2: number;

    constructor(batch: number, nh: number, T1: number, T2: number, hs: number) {
        this.shaderKey = `AttentionMask_${hs}`;
        this.outputShape = [batch, nh, T1, T2 / 2];
        this.hs = hs;
        this.nh = nh;
        this.T1 = T1;
        this.T2 = T2;

        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [0.5, 1, 1]);

        if (hs % 4 !== 0) {
            throw new Error('Head size must be a multiple of 4 for AttentionMaskProgram');
        }
    }

    getUserCode(): string {
        const userCode = `
            var<workgroup> outShared : array<f32, ${this.workgroupSize[0] / 2}>;
            ${main('index')} {
                
                let coords = getCoordsFromIndex(index / 2);
                let b = coords[0];
                let h = coords[1];
                let t1 = coords[2];
                let t2 = coords[3] * 2i + i32(localId.x % 2u);

                    var sum: f32 = 0.0f;

                    if (t2 > t1 + uniforms.pastLen) {
                        sum = uniforms.inf;
                    } else {
                        let q0 = getIndexFromCoords4D(vec4<i32>(b, h, t1, 0), uniforms.qShape);
                        let k0 = getIndexFromCoords4D(vec4<i32>(b, h, t2, 0), uniforms.kShape);
                        
                        
                        for (var i: i32 = 0; i < ${this.hs}; i = i + 4) {
                            let qv = vec4<i32>(q[q0 + i], q[q0 + i + 1], q[q0 + i + 2], q[q0 + i + 3]);
                            let kv = vec4<i32>(k[k0 + i], k[k0 + i + 1], k[k0 + i + 2], k[k0 + i + 3]);

                            //sum = sum + dot(unpack2x16float(u32(qv[0])), unpack2x16float(u32(kv[0])));
                            //sum = sum + dot(unpack2x16float(u32(qv[1])), unpack2x16float(u32(kv[1])));
                            //sum = sum + dot(unpack2x16float(u32(qv[2])), unpack2x16float(u32(kv[2])));
                            //sum = sum + dot(unpack2x16float(u32(qv[3])), unpack2x16float(u32(kv[3])));

                            sum = sum + dot(vec4<f32>(unpack2x16float(u32(qv[0])), unpack2x16float(u32(qv[1]))),
                                            vec4<f32>(unpack2x16float(u32(kv[0])), unpack2x16float(u32(kv[1]))));
                            sum = sum + dot(vec4<f32>(unpack2x16float(u32(qv[2])), unpack2x16float(u32(qv[3]))),
                                            vec4<f32>(unpack2x16float(u32(kv[2])), unpack2x16float(u32(kv[3]))));
                        }
                        sum = sum * uniforms.divisor;
                    }

                    if (localId.x % 2 == 1) {
                        outShared[localId.x >> 1] = sum;
                    }
                    // Probably not needed since other thread is always in same subgroup
                    // Verify this later! Or use subgroup quad operations
                    workgroupBarrier();

                    if (localId.x % 2 == 0) {
                        result[index / 2] = i32(pack2x16float(vec2<f32>(sum, outShared[localId.x >> 1])));
                    }
            }
        `;
        return userCode;
    }
}

function attentionMaskGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { q, k } = args.inputs as { q: Tensor; k: Tensor };
    const { divisor, pastLen } = args.attrs as { divisor: number; pastLen: number };

    const backend = args.backend as WebGPUBackend;

    const packed = isPackedTensor(q) && isPackedTensor(k);

    const batchSize = q.shape[0];
    const T1 = q.shape[2]!; // Sequence length
    const T2 = k.shape[2]!; // Sequence length
    const nh = q.shape[1]!; // Number of heads
    const hs = q.shape[3]!; // Head size

    assertShapesMatch(k.shape, [batchSize, nh, T2, hs], 'Error in AttentionMask: ');
    if (divisor === 0) {
        throw new Error('Divisor must be non-zero in AttentionMask');
    }
    if (pastLen < 0) {
        throw new Error('pastLen must be non-negative in AttentionMask');
    }

    const program = packed
        ? new AttentionMaskProgram16(batchSize, nh, T1, T2, hs)
        : new AttentionMaskProgram32(batchSize, nh, T1, T2, hs);
    const uniformData = [
        { type: 'float32', data: [divisor] },
        { type: 'int32', data: [pastLen] },
        { type: 'float32', data: [Number.NEGATIVE_INFINITY] },
    ];

    const dtype = packed ? 'int32' : q.dtype;
    const result: PackedTensorInfo = backend.runWebGPUProgram(program, [q, k], dtype, uniformData);
    result.packed = packed;
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'AttentionMask',
    backendName: 'webgpu',
    kernelFunc: attentionMaskGPU,
};

registerKernel(kernelConfig);
