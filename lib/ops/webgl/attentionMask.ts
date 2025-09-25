import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';

class AttentionMaskProgram implements GPGPUProgram {
    variableNames = ['q', 'k'];
    outputShape: number[];
    userCode: string;
    customUniforms = [
        { name: 'divisor', type: 'float' as UniformType },
        { name: 'pastLen', type: 'int' as UniformType },
        { name: 'inf', type: 'float' as UniformType },
    ];

    constructor(batch: number, nh: number, T1: number, T2: number, hs: number) {
        this.outputShape = [batch, nh, T1, T2];

        this.userCode = `
        void main() {
            ivec4 coords = getOutputCoords(); // [batch, nh, t1, t2]
            int b = coords.x;
            int h = coords.y;
            int t1 = coords.z;
            int t2 = coords.w;

            float sum = 0.0;
            for (int i = 0; i < ${hs}; ++i) {
                float qv = getQ(b, h, t1, i);
                float kv = getK(b, h, t2, i);
                sum += qv * kv;
            }

            // Scale by divisor
            float scaled = sum * divisor;

            // Mask out future positions
            setOutput((t2 > t1 + pastLen) ? inf : scaled);
        }
        `;
    }
}

function attentionMaskGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { q, k } = args.inputs as { q: Tensor; k: Tensor };
    const { divisor, pastLen } = args.attrs as { divisor: number; pastLen: number };

    const backend = args.backend as MathBackendWebGL;

    const batchSize = q.shape[0];
    const T1 = q.shape[2]!; // Sequence length
    const T2 = k.shape[2]!; // Sequence length
    const nh = q.shape[1]!; // Number of heads
    const hs = q.shape[3]!; // Head size

    const program = new AttentionMaskProgram(batchSize, nh, T1, T2, hs);
    return backend.runWebGLProgram(program, [q, k], 'float32', [[divisor], [pastLen], [Number.NEGATIVE_INFINITY]]);
}

const kernelConfig: KernelConfig = {
    kernelName: 'AttentionMask',
    backendName: 'webgl',
    kernelFunc: attentionMaskGPU,
};

registerKernel(kernelConfig);
