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
    variableNames = ['q', 'k', 'mask'];
    outputShape: number[];
    userCode: string;
    customUniforms = [{ name: 'divisor', type: 'float' as UniformType }];

    constructor(batch: number, nh: number, T: number, hs: number) {
        this.outputShape = [batch, nh, T, T];

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
                float kv = getK(b, h, t2, i); // k is transposed on last two dims
                sum += qv * kv;
            }

            // Scale by divisor
            float scaled = sum * divisor;

            // Add mask
            float maskVal = getMask(t1, t2); // mask is [T,T]

            setOutput(scaled + maskVal);
        }
        `;
    }
}

function attentionMaskGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { q, k, mask } = args.inputs as { q: Tensor; k: Tensor; mask: Tensor };
    const { divisor } = args.attrs as { divisor: number };

    const backend = args.backend as MathBackendWebGL;

    const batchSize = q.shape[0];
    const T = q.shape[2]!; // Sequence length
    const nh = q.shape[1]!; // Number of heads

    const program = new AttentionMaskProgram(batchSize, nh, T, q.shape[3]!);
    return backend.runWebGLProgram(program, [q, k, mask], 'float32', [[divisor]]);
}

const kernelConfig: KernelConfig = {
    kernelName: 'AttentionMask',
    backendName: 'webgl',
    kernelFunc: attentionMaskGPU,
};

registerKernel(kernelConfig);
