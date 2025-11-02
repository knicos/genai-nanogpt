import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';
import { registerKernel, KernelConfig, TensorInfo, NamedTensorInfoMap, NamedAttrMap } from '@tensorflow/tfjs-core';

class AdamMomentsProgram implements GPGPUProgram {
    variableNames = ['moments', 'gradient'];
    outputShape: number[];
    userCode: string;
    customUniforms = [
        { name: 'beta1', type: 'float' as UniformType },
        { name: 'beta2', type: 'float' as UniformType },
    ];

    constructor(outputShape: number[]) {
        this.outputShape = outputShape;

        const rank = outputShape.length;
        const coordsType = rank === 1 ? 'int' : `ivec${Math.min(rank, 4)}`;
        const lastCoordExpr = rank === 1 ? 'coords' : `coords[${rank - 1}]`;
        const coordArgs =
            rank === 1
                ? 'coords'
                : outputShape
                      .slice(0, -1)
                      .map((_, i) => `coords[${i}]`)
                      .join(', ');

        this.userCode = `
        void main() {
            float m = getMomentsAtOutCoords();
            ${coordsType} coords = getOutputCoords();

            // Add gradient clipping here
            float g = clamp(getGradient(${coordArgs}), -1.0, 1.0);
            int which = ${lastCoordExpr};

            float beta = which == 0 ? beta1 : beta2;
            float gg = which == 0 ? g : g * g;
    
            float newM = m * beta + gg * (1.0 - beta);
            setOutput(newM);
        }
        `;
    }
}

function adamMomentsGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { moments, gradient } = args.inputs as { moments: TensorInfo; gradient: TensorInfo };
    const { beta1, beta2 } = args.attrs as { beta1: number; beta2: number };

    const backend = args.backend as MathBackendWebGL;

    const program = new AdamMomentsProgram(moments.shape);
    return backend.runWebGLProgram(program, [moments, gradient], 'float32', [[beta1], [beta2]]);
}

const kernelConfig: KernelConfig = {
    kernelName: 'AdamMoments',
    backendName: 'webgl',
    kernelFunc: adamMomentsGPU,
};

registerKernel(kernelConfig);
