import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { registerKernel, KernelConfig, TensorInfo, NamedTensorInfoMap, NamedAttrMap } from '@tensorflow/tfjs-core';

class AdamMomentsProgram implements WebGPUProgram {
    variableNames = ['moments', 'gradient'];
    outputShape: number[];
    shaderKey = 'AdamMoments';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    uniforms = 'beta1: f32, beta2: f32';
    outputComponent = 2;
    variableComponents = [2, 1];

    constructor(outputShape: number[]) {
        this.outputShape = outputShape;

        this.dispatchLayout = flatDispatchLayout(this.outputShape.slice(0, -1));
        this.dispatch = computeDispatch(
            this.dispatchLayout,
            this.outputShape.slice(0, -1),
            this.workgroupSize,
            [1, 1, 1]
        );
    }
    getUserCode() {
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let m: vec2<f32> = moments[index];
                let g: f32 = gradient[index];

                let newM1 = fma(m.x, uniforms.beta1, g * (1.0 - uniforms.beta1));
                let newM2 = fma(m.y, uniforms.beta2, g * g * (1.0 - uniforms.beta2));

                setOutputAtIndex(index, vec2<f32>(newM1, newM2));
            }
        }
    `;
    }
}

function adamMomentsGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { moments, gradient } = args.inputs as { moments: TensorInfo; gradient: TensorInfo };
    const { beta1, beta2 } = args.attrs as { beta1: number; beta2: number };

    const backend = args.backend as WebGPUBackend;

    const program = new AdamMomentsProgram(moments.shape);
    const uniformData = [
        { type: 'float32', data: [beta1] },
        { type: 'float32', data: [beta2] },
    ];
    return backend.runWebGPUProgram(program, [moments, gradient], 'float32', uniformData);
}

const kernelConfig: KernelConfig = {
    kernelName: 'AdamMoments',
    backendName: 'webgpu',
    kernelFunc: adamMomentsGPU,
};

registerKernel(kernelConfig);
