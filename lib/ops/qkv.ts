import { GPGPUProgram, MathBackendWebGL, Tensor, engine } from '@tensorflow/tfjs';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    registerGradient,
    GradConfig,
    reshape,
    split,
} from '@tensorflow/tfjs-core';

class QKVProgram implements GPGPUProgram {
    variableNames = ['x', 'kernel'];
    outputShape: number[];
    userCode: string;
    // enableShapeUniforms = true;
    customUniforms = [{ name: 'mode', type: 'int' as UniformType }];

    constructor(batch: number, nh: number, T: number, C: number) {
        // Output shape: [batch, nh, T, T]
        const head_dim = C / nh;
        this.outputShape = [batch, nh, T, head_dim];

        this.userCode = `
        void main() {
            ivec4 coords = getOutputCoords(); // [b, h, t, d]
            int b = coords.x;
            int h = coords.y;
            int t = coords.z;
            int d = coords.w;

            // Compute output channel index in fused kernel
            int out_offset = mode * ${nh} * ${head_dim} + h * ${head_dim} + d;

            float sum = 0.0;
            for (int c = 0; c < ${C}; ++c) {
                float xval = getX(b, t, c); // fetch from x
                float kval = getKernel(c, out_offset); // fetch from kernel
                sum += xval * kval;
            }

            setOutput(sum);
        }
        `;
    }
}

function qkvGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo[] {
    const { x, kernel } = args.inputs as { x: Tensor; kernel: Tensor };
    const { heads } = args.attrs as { heads: number };

    const backend = args.backend as MathBackendWebGL;

    const batchSize = x.shape[0];
    const seqLength = x.shape[1]!;
    const C = x.shape[2]!;

    const program = new QKVProgram(batchSize, heads, seqLength, C);
    return [
        backend.runWebGLProgram(program, [x, kernel], 'float32', [[0]]),
        backend.runWebGLProgram(program, [x, kernel], 'float32', [[1]]),
        backend.runWebGLProgram(program, [x, kernel], 'float32', [[2]]),
    ];
}

const kernelConfig: KernelConfig = {
    kernelName: 'QKV',
    backendName: 'webgl',
    kernelFunc: qkvGPU,
};

registerKernel(kernelConfig);

// CPU fallback implementation
export function qkvCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo[] {
    const { x, kernel } = args.inputs as { x: Tensor; kernel: Tensor };
    const { heads } = args.attrs as { heads: number };

    const [B, T, C] = x.shape; // batch size, sequence length, embedding dimensionality

    // Calculate query, key, values for all heads in batch and move head forward to be the batch dim
    const x2d = reshape(x, [B * T, C]);
    const qkvFlat = x2d.dot(kernel); //this.cAttn.apply(x) as TF.Tensor; // (B, T, 3*C)
    //x.dispose();
    x2d.dispose();
    const qkv = reshape(qkvFlat, [B, T, 3 * C]);
    qkvFlat.dispose();

    const [q, k, v] = split(qkv, 3, -1); // Each is (B, T, C)
    qkv.dispose();

    // Reshape for multi-head attention
    const headDim = C / heads;

    const qReshaped = reshape(q, [B, T, heads, headDim]);
    q.dispose();
    const qT = qReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
    qReshaped.dispose();

    const kReshaped = reshape(k, [B, T, heads, headDim]);
    k.dispose();
    const kT = kReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
    kReshaped.dispose();

    const vReshaped = reshape(v, [B, T, heads, headDim]);
    v.dispose();
    const vT = vReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
    vReshaped.dispose();

    return [qT, kT, vT];
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'QKV',
    backendName: 'cpu',
    kernelFunc: qkvCPU,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'QKV',
    backendName: 'tensorflow',
    kernelFunc: qkvCPU,
};

registerKernel(tensorflowKernelConfig);

export function qkv(x: Tensor, kernel: Tensor, heads: number): Tensor[] {
    return engine().runKernel('QKV', { x, kernel }, { heads });
}

const qkvGradConfig: GradConfig = {
    kernelName: 'QKV',
    inputsToSave: ['x', 'kernel'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        // dy: [dq, dk, dv]
        // x: input tensor
        // kernel: fused kernel weights
        const [dq, dk, dv] = dy as Tensor[];
        const [x, kernel] = saved as Tensor[];

        // Get shapes
        const [B, T, C] = x.shape;

        // Reshape dy to [B*T, C] for each Q, K, V
        const dq2d = dq.transpose([0, 2, 1, 3]).reshape([B * T, C]);
        const dk2d = dk.transpose([0, 2, 1, 3]).reshape([B * T, C]);
        const dv2d = dv.transpose([0, 2, 1, 3]).reshape([B * T, C]);

        // Gradient w.r.t x: sum of matMul with each kernel slice
        const kernelQ = kernel.slice([0, 0], [C, C]);
        const kernelK = kernel.slice([0, C], [C, C]);
        const kernelV = kernel.slice([0, 2 * C], [C, C]);

        return {
            x: () => {
                const dxQ = dq2d.matMul(kernelQ, false, true);
                const dxK = dk2d.matMul(kernelK, false, true);
                const dxV = dv2d.matMul(kernelV, false, true);
                const dx = dxQ.add(dxK).add(dxV).reshape([B, T, C]);
                return dx;
            },
            kernel: () => {
                // Gradient w.r.t kernel: x^T * dy for each Q, K, V
                const x2d = x.reshape([B * T, C]);
                const dKernelQ = x2d.matMul(dq2d, true, false);
                const dKernelK = x2d.matMul(dk2d, true, false);
                const dKernelV = x2d.matMul(dv2d, true, false);
                const dKernel = dKernelQ.concat(dKernelK, 1).concat(dKernelV, 1); // [C, 3*C]
                return dKernel;
            },
        };
    },
};

registerGradient(qkvGradConfig);
