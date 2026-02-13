import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { registerKernel, KernelConfig, TensorInfo, NamedTensorInfoMap, NamedAttrMap } from '@tensorflow/tfjs-core';
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';

class AdamAdjustProgram implements WebGPUProgram {
    variableNames = ['moments', 'value'];
    outputShape: number[];
    shaderKey = 'AdamAdjust';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    uniforms = 'invbeta1: f32, invbeta2: f32, learningRate: f32, epsilon: f32';
    outputComponent = 1;
    variableComponents = [2, 1];
    useWeightDecay: boolean;

    constructor(outputShape: number[], useWeightDecay: boolean) {
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.useWeightDecay = useWeightDecay;
        if (useWeightDecay) {
            this.uniforms += ', weightDecay: f32';
        }
    }
    getUserCode() {
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let moments: vec2<f32> = moments[index];
                let value: f32 = value[index];

                let m1Hat = moments.x * uniforms.invbeta1;
                let m2Hat = moments.y * uniforms.invbeta2;

                let invSqrt = inverseSqrt(max(m2Hat, 1e-30));
                let invDenom = invSqrt / fma(uniforms.epsilon, invSqrt, 1.0);
                var adjustedValue = fma(-uniforms.learningRate * m1Hat, invDenom, value);

                ${this.useWeightDecay ? 'adjustedValue = adjustedValue - uniforms.learningRate * uniforms.weightDecay * value;' : ''}

                setOutputAtIndex(index, adjustedValue);
            }
        }
    `;
    }
}

function adamAdjustGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { moments, value } = args.inputs as { moments: TensorInfo; value: TensorInfo };
    const { beta1, beta2, learningRate, epsilon, weightDecay } = args.attrs as {
        beta1: number;
        beta2: number;
        learningRate: number;
        epsilon: number;
        weightDecay: number;
    };

    const backend = args.backend as WebGPUBackend;

    assertShapesMatch(moments.shape, [...value.shape, 2], 'Error in AdamAdjust: ');

    const program = new AdamAdjustProgram(value.shape, weightDecay > 0);
    const uniformData = [
        { type: 'float32', data: [1.0 / beta1] },
        { type: 'float32', data: [1.0 / beta2] },
        { type: 'float32', data: [learningRate] },
        { type: 'float32', data: [epsilon] },
    ];
    if (weightDecay > 0) {
        uniformData.push({ type: 'float32', data: [weightDecay] });
    }
    return backend.runWebGPUProgram(program, [moments, value], 'float32', uniformData);
}

const kernelConfig: KernelConfig = {
    kernelName: 'AdamAdjust',
    backendName: 'webgpu',
    kernelFunc: adamAdjustGPU,
};

registerKernel(kernelConfig);
