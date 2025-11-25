import {
    KernelConfig,
    KernelFunc,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
} from '@tensorflow/tfjs-core';
import { unaryKernelFunc } from '@tensorflow/tfjs-backend-webgl/dist/kernel_utils/kernel_funcs_utils';
import { CHECK_NAN_SNIPPET } from '@tensorflow/tfjs-backend-webgl/dist/unaryop_gpu';
import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';

const K = 0.7978845608028654; // sqrt(2/pi)
const A = 0.044715;

const GELU =
    CHECK_NAN_SNIPPET +
    `
    float x3 = x * x * x;
    float inner = x + ${A} * x3;
    inner = ${K} * inner;
    inner = tanh(inner);
    inner = 0.5 * (1.0 + inner);
    inner = x * inner;
    return inner;
`;

export const gelu = unaryKernelFunc({ opSnippet: GELU });

const geluConfig: KernelConfig = {
    kernelName: 'Gelu',
    backendName: 'webgl',
    kernelFunc: gelu as unknown as KernelFunc,
};

registerKernel(geluConfig);

// Backward

class GeluGradProgram implements GPGPUProgram {
    // Inputs: dy, x
    variableNames = ['dy', 'x'];
    outputShape: number[];
    userCode: string;

    constructor(shape: number[]) {
        this.outputShape = shape;
        // d/dx gelu(x) = 0.5*(1 + t) + 0.5*x*(1 - t^2)*k*(1 + 3a x^2)
        this.userCode = `
            void main() {
                float dy = getDyAtOutCoords();
                float x  = getXAtOutCoords();
                float x2 = x * x;
                float x3 = x2 * x;
                float u  = ${K} * (x + ${A} * x3);
                float t  = tanh(u);
                float sech2 = 1.0 - t * t;
                float du_dx = ${K} * (1.0 + 3.0 * ${A} * x2);
                float dgelu = 0.5 * (1.0 + t) + 0.5 * x * sech2 * du_dx;
                setOutput(dy * dgelu);
            }`;
    }
}

// Backward kernel
function geluGradKernelFunc(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { dy, x } = args.inputs as { dy: Tensor; x: Tensor };
    const backend = args.backend as MathBackendWebGL;
    const program = new GeluGradProgram(x.shape);
    return backend.runWebGLProgram(program, [dy, x], 'float32');
}

const geluGradKernelConfig: KernelConfig = {
    kernelName: 'GeluGrad',
    backendName: 'webgl',
    kernelFunc: geluGradKernelFunc,
};

registerKernel(geluGradKernelConfig);
