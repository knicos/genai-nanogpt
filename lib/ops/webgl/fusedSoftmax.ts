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

    constructor(batch: number, nh: number, T: number) {
        this.outputShape = [batch, nh, T, T];

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

    constructor(batch: number, nh: number, T: number) {
        this.outputShape = [batch, nh, T, T];
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

    //const maxLogitsReshaped = reshape(maxLogit, expandedShape);
    //const a = sub(logits as Tensor, maxLogitsReshaped);
    //const b = exp(a);

    const batchSize = logits.shape[0];
    const T = logits.shape[2]!; // Sequence length
    const nh = logits.shape[1]!; // Number of heads

    // Use the SubExpProgram to compute exp(logits - maxLogit) in one shader
    // program, rather than doing it in two steps.
    const subExp = new SubExpProgram(batchSize, nh, T);
    const expTensor = backend.runWebGLProgram(subExp, [logits, maxLogit], 'float32');

    const sumExp = sum({ inputs: { x: expTensor }, backend, attrs: { axis: axes, keepDims: false } });
    const sumExpReshaped = reshape({ inputs: { x: sumExp }, backend, attrs: { shape: expandedShape } });

    /*const $noiseShape = getNoiseShape($x, noiseShape);
    const keepProb = 1 - rate;
    const multiplier = div(
        floor(add(randomUniform($noiseShape, 0, 1, 'float32', seed), keepProb)),
        keepProb);

    return mul($x, multiplier);*/

    if (dropoutRate !== undefined && dropoutRate > 0) {
        const dropoutProg = new DivDropoutProgram(batchSize, nh, T);
        const out = backend.runWebGLProgram(dropoutProg, [expTensor, sumExpReshaped], 'float32', [
            [dropoutRate],
            [seed ?? Math.random() * 10000],
        ]);
        backend.disposeIntermediateTensorInfo(maxLogit);
        backend.disposeIntermediateTensorInfo(expTensor);
        backend.disposeIntermediateTensorInfo(sumExp);
        backend.disposeIntermediateTensorInfo(sumExpReshaped);
        return out;
    }

    const res = realDiv({ inputs: { a: expTensor, b: sumExpReshaped }, backend }) as TensorInfo;

    backend.disposeIntermediateTensorInfo(maxLogit);
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
