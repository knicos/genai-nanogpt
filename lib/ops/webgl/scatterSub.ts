import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { registerKernel, KernelConfig, TensorInfo, NamedTensorInfoMap } from '@tensorflow/tfjs-core';

class SoftmaxSubOneHotProgram implements GPGPUProgram {
    variableNames = ['labels', 'softmaxProbs', 'dy'];
    outputShape: number[];
    userCode: string;

    constructor(batchSize: number, depth: number) {
        this.outputShape = [batchSize, depth];

        this.userCode = `
            void main() {
                ivec2 coords = getOutputCoords();
                int index = int(getLabels(coords.x));
                float prob = getSoftmaxProbsAtOutCoords();
                float dy = getDy(coords.x);
                setOutput((index == coords.y ? prob - 1.0 : prob) * dy);
            }
            `;
    }
}

function efficientScatterSub(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { logits, labels, dy } = args.inputs as { logits: TensorInfo; labels: TensorInfo; dy: TensorInfo };

    const backend = args.backend as MathBackendWebGL;

    const batchSize = labels.shape[0];
    const depth = logits.shape[1];

    const program = new SoftmaxSubOneHotProgram(batchSize, depth);
    return backend.runWebGLProgram(program, [labels, logits, dy], 'float32');
}

const kernelConfig: KernelConfig = {
    kernelName: 'EfficientScatterSub',
    backendName: 'webgl',
    kernelFunc: efficientScatterSub,
};

registerKernel(kernelConfig);
