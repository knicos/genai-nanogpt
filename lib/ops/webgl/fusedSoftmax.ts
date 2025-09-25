import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { max } from '@tensorflow/tfjs-backend-webgl/dist/kernels/Max';
import { sum } from '@tensorflow/tfjs-backend-webgl/dist/kernels/Sum';
import { realDiv } from '@tensorflow/tfjs-backend-webgl/dist/kernels/RealDiv';
import { reshape } from '@tensorflow/tfjs-backend-webgl/dist/kernels/Reshape';
import {
    backend_util,
    KernelConfig,
    KernelFunc,
    registerKernel,
    SoftmaxAttrs,
    SoftmaxInputs,
    TensorInfo,
    util,
} from '@tensorflow/tfjs-core';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';

class SubExpProgram implements GPGPUProgram {
    variableNames = ['logits', 'maxLogits'];
    outputShape: number[];
    userCode: string;

    constructor(shape: number[]) {
        this.outputShape = shape;

        this.userCode = `
        void main() {
        ivec4 coords = getOutputCoords(); // [batch, nh, t1, t2]
            int b = coords.x;
            int h = coords.y;
            int t1 = coords.z;
            int t2 = coords.w;
            float x = getLogitsAtOutCoords();
            float maxLogit = getMaxLogits(b, h, t1);
            setOutput(exp(x - maxLogit));
        }
        `;
    }
}

class DivDropoutProgram implements GPGPUProgram {
    variableNames = ['exp', 'sum'];
    outputShape: number[];
    userCode: string;
    customUniforms = [
        { name: 'dropoutRate', type: 'float' as UniformType },
        { name: 'seed', type: 'float' as UniformType },
    ];

    constructor(shape: number[]) {
        this.outputShape = shape;
        this.userCode = `
        float random(ivec4 coords) {
            float x = float(coords.x * 4096 + coords.y * 256 + coords.z * 16 + coords.w);
            return fract(sin(seed + x) * 43758.5453123);
        }
        void main() {
            ivec4 coords = getOutputCoords();
            float numerator = getExp(coords.x, coords.y, coords.z, coords.w);
            float denominator = getSum(coords.x, coords.y, coords.z, coords.w);
            float val = numerator / denominator;
            float keepProb = 1.0 - dropoutRate;
            float rand = random(coords);
            float mask = step(rand, keepProb);
            setOutput(val * mask / keepProb);
        }
        `;
    }
}

interface FusedSoftmaxAttrs extends SoftmaxAttrs {
    dropoutRate?: number;
    seed?: number;
}

export function softmax(args: { inputs: SoftmaxInputs; backend: unknown; attrs: FusedSoftmaxAttrs }): TensorInfo {
    const { inputs, attrs } = args;
    const { logits } = inputs;
    const { dim, dropoutRate, seed } = attrs;
    const backend = args.backend as MathBackendWebGL;

    if (!logits) {
        throw new Error('Error in softmax: input logits is null');
    }

    const axes = util.parseAxisParam([dim], logits.shape);

    const maxLogit = max({
        inputs: { x: logits },
        backend,
        attrs: { reductionIndices: axes, keepDims: false },
    });

    const expandedShape = backend_util.expandShapeToKeepDim(maxLogit.shape, axes);

    // Use the SubExpProgram to compute exp(logits - maxLogit) in one shader
    // program, rather than doing it in two steps.
    const subExp = new SubExpProgram(logits.shape);
    const expTensor = backend.runWebGLProgram(subExp, [logits, maxLogit], 'float32');
    backend.disposeIntermediateTensorInfo(maxLogit);

    const sumExp = sum({ inputs: { x: expTensor }, backend, attrs: { axis: axes, keepDims: false } });
    const sumExpReshaped = reshape({ inputs: { x: sumExp }, backend, attrs: { shape: expandedShape } });

    if (dropoutRate !== undefined && dropoutRate > 0) {
        const dropoutProg = new DivDropoutProgram(logits.shape);
        const out = backend.runWebGLProgram(dropoutProg, [expTensor, sumExpReshaped], 'float32', [
            [dropoutRate],
            [seed ?? Math.random() * 10000],
        ]);

        backend.disposeIntermediateTensorInfo(expTensor);
        backend.disposeIntermediateTensorInfo(sumExp);
        backend.disposeIntermediateTensorInfo(sumExpReshaped);
        return out;
    }

    const res = realDiv({ inputs: { a: expTensor, b: sumExpReshaped }, backend }) as TensorInfo;

    backend.disposeIntermediateTensorInfo(expTensor);
    backend.disposeIntermediateTensorInfo(sumExp);
    backend.disposeIntermediateTensorInfo(sumExpReshaped);

    return res;
}

const webglKernelConfig: KernelConfig = {
    kernelName: 'FusedSoftmax',
    backendName: 'webgl',
    kernelFunc: softmax as unknown as KernelFunc,
};

registerKernel(webglKernelConfig);
