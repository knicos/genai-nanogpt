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

class AttentionMaskProgram implements WebGPUProgram {
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
                    
                    var sum: f32 = 0.0;
                    for (var i: i32 = 0; i < ${this.hs}; i = i + 4) {
                        let q0 = getIndexFromCoords4D(vec4<i32>(b, h, t1, i), uniforms.qShape);
                        let qv = vec4<f32>(q[q0], q[q0 + 1], q[q0 + 2], q[q0 + 3]);
                        let k0 = getIndexFromCoords4D(vec4<i32>(b, h, t2, i), uniforms.kShape);
                        let kv = vec4<f32>(k[k0], k[k0 + 1], k[k0 + 2], k[k0 + 3]);
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

function attentionMaskGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { q, k } = args.inputs as { q: Tensor; k: Tensor };
    const { divisor, pastLen } = args.attrs as { divisor: number; pastLen: number };

    const backend = args.backend as WebGPUBackend;

    const batchSize = q.shape[0];
    const T1 = q.shape[2]!; // Sequence length
    const T2 = k.shape[2]!; // Sequence length
    const nh = q.shape[1]!; // Number of heads
    const hs = q.shape[3]!; // Head size

    const program = new AttentionMaskProgram(batchSize, nh, T1, T2, hs);
    const uniformData = [
        { type: 'float32', data: [divisor] },
        { type: 'int32', data: [pastLen] },
        { type: 'float32', data: [Number.NEGATIVE_INFINITY] },
    ];

    return backend.runWebGPUProgram(program, [q, k], 'float32', uniformData);
}

const kernelConfig: KernelConfig = {
    kernelName: 'AttentionMask',
    backendName: 'webgpu',
    kernelFunc: attentionMaskGPU,
};

registerKernel(kernelConfig);
