import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { KernelConfig, KernelFunc, registerKernel, TensorInfo } from '@tensorflow/tfjs-core';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';

class MulDropoutProgram implements GPGPUProgram {
    variableNames = ['a', 'b'];
    outputShape: number[];
    userCode: string;
    customUniforms = [
        { name: 'dropoutRate', type: 'float' as UniformType },
        { name: 'seed', type: 'float' as UniformType },
    ];

    constructor(batch: number, nh: number, T: number) {
        this.outputShape = [batch, nh, T, T];
        this.userCode = `
        float random(ivec4 coords) {
            float x = float(coords.x * 4096 + coords.y * 256 + coords.z * 16 + coords.w);
            return fract(sin(seed + x) * 43758.5453123);
        }
        void main() {
            ivec4 coords = getOutputCoords();
            float a = getA(coords.x, coords.y, coords.z, coords.w);
            float b = getB(coords.x, coords.y, coords.z, coords.w);
            
            float keepProb = 1.0 - dropoutRate;
            float rand = random(coords);
            float mask = step(rand, keepProb);
            setOutput(a * b * mask / keepProb);
        }
        `;
    }
}

interface MulDropoutAttrs {
    dropoutRate?: number;
    seed?: number;
}

function mulDrop(args: {
    inputs: { a: TensorInfo; b: TensorInfo };
    backend: unknown;
    attrs: MulDropoutAttrs;
}): TensorInfo {
    const { inputs, attrs } = args;
    const { a, b } = inputs;
    const { dropoutRate, seed } = attrs;
    const backend = args.backend as MathBackendWebGL;

    const batchSize = a.shape[0];
    const T = a.shape[2]!; // Sequence length
    const nh = a.shape[1]!; // Number of heads

    // Use the SubExpProgram to compute exp(logits - maxLogit) in one shader
    // program, rather than doing it in two steps.
    const dropProgram = new MulDropoutProgram(batchSize, nh, T);
    const res = backend.runWebGLProgram(dropProgram, [a, b], 'float32', [
        [dropoutRate ?? 0],
        [seed ?? Math.random() * 10000],
    ]);
    return res;
}

const webglKernelConfig: KernelConfig = {
    kernelName: 'MulDropout',
    backendName: 'webgl',
    kernelFunc: mulDrop as unknown as KernelFunc,
};

registerKernel(webglKernelConfig);
