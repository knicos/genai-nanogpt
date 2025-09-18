import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import {
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';

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
