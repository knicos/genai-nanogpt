import { GPGPUProgram, MathBackendWebGL, Tensor, engine } from '@tensorflow/tfjs';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    matMul,
    scalar,
    NamedAttrMap,
    registerGradient,
    GradConfig,
} from '@tensorflow/tfjs-core';

class AttentionMaskProgram implements GPGPUProgram {
    variableNames = ['q', 'k', 'mask'];
    outputShape: number[];
    userCode: string;
    // enableShapeUniforms = true;
    customUniforms = [{ name: 'divisor', type: 'float' as UniformType }];

    constructor(batch: number, nh: number, T: number, hs: number) {
        // Output shape: [batch, nh, T, T]
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

// CPU fallback implementation
function attentionMaskCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo {
    const { q, k, mask } = args.inputs as { q: Tensor; k: Tensor; mask: Tensor };
    const { divisor } = args.attrs as { divisor: number };

    const T = q.shape[2]!; // Sequence length

    // Causal self-attention
    const attUnscaled = matMul(q, k, false, true); // (B, nh, T, T)
    const att = attUnscaled.mul(scalar(divisor)); // Scale by sqrt(d_k)
    const mask2 = mask.slice([0, 0], [T, T]).expandDims(0).expandDims(0); // (1,1,T,T)
    const maskedAtt = att.add(mask2);
    return maskedAtt;
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'AttentionMask',
    backendName: 'cpu',
    kernelFunc: attentionMaskCPU,
};

registerKernel(cpuKernelConfig);

export function attentionMask(q: Tensor, k: Tensor, mask: Tensor, divisor: number): Tensor {
    return engine().runKernel('AttentionMask', { q, k, mask }, { divisor });
}

const attentionMaskGradConfig: GradConfig = {
    kernelName: 'AttentionMask',
    inputsToSave: ['q', 'k'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[], attrs: NamedAttrMap) => {
        if (Array.isArray(dy)) {
            throw new Error('Expected dy to be a single Tensor');
        }
        const [q, k] = saved as Tensor[];
        const { divisor } = attrs as { divisor: number };

        return {
            q: () => dy.matMul(k).mul(divisor),
            k: () => q.transpose([0, 1, 3, 2]).matMul(dy).mul(divisor).transpose([0, 1, 3, 2]),
            mask: () => dy,
            divisor: () => {
                const attUnscaled = q.matMul(k, false, true);
                return dy.mul(attUnscaled).sum();
            },
        };
    },
};

registerGradient(attentionMaskGradConfig);
