import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { reshape } from '@tensorflow/tfjs-backend-webgl/dist/kernels/Reshape';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';

class AdamAdjustProgram implements GPGPUProgram {
    variableNames = ['moments', 'value'];
    outputShape: number[];
    userCode: string;
    customUniforms = [
        { name: 'invBeta1', type: 'float' as UniformType },
        { name: 'invBeta2', type: 'float' as UniformType },
        { name: 'learningRate', type: 'float' as UniformType },
        { name: 'epsilon', type: 'float' as UniformType },
    ];

    constructor(outputShape: number[]) {
        this.outputShape = outputShape;

        this.userCode = `
        void main() {
            float v = getValueAtOutCoords();
            int coords = getOutputCoords();
            coords *= 2;
            float m1 = getMoments(coords);
            float m2 = getMoments(coords + 1);

            float m1Hat = m1 * invBeta1;
            float m2Hat = m2 * invBeta2;

            float invSqrt = inversesqrt(max(m2Hat, 1e-30));
            float invDenom = invSqrt / (1.0 + epsilon * invSqrt);
            float adjustedValue = -learningRate * m1Hat * invDenom + v;

            setOutput(adjustedValue);
        }
        `;
    }
}

function adamAdjustGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { moments, value } = args.inputs as { moments: Tensor; value: Tensor };
    const { beta1, beta2, learningRate, epsilon } = args.attrs as {
        beta1: number;
        beta2: number;
        learningRate: number;
        epsilon: number;
    };

    const backend = args.backend as MathBackendWebGL;

    const momentsReshaped = reshape({ inputs: { x: moments }, backend, attrs: { shape: [-1] } });
    const valueReshaped = reshape({ inputs: { x: value }, backend, attrs: { shape: [-1] } });

    const program = new AdamAdjustProgram(valueReshaped.shape);
    const result = backend.runWebGLProgram(program, [momentsReshaped, valueReshaped], 'float32', [
        [1 / beta1],
        [1 / beta2],
        [learningRate],
        [epsilon],
    ]);

    backend.disposeIntermediateTensorInfo(momentsReshaped);
    backend.disposeIntermediateTensorInfo(valueReshaped);
    const resultReshaped = reshape({ inputs: { x: result }, backend, attrs: { shape: value.shape } });
    backend.disposeIntermediateTensorInfo(result);
    return resultReshaped;
}

const kernelConfig: KernelConfig = {
    kernelName: 'AdamAdjust',
    backendName: 'webgl',
    kernelFunc: adamAdjustGPU,
};

registerKernel(kernelConfig);
