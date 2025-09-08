import { GPGPUProgram, MathBackendWebGL, Tensor, engine } from '@tensorflow/tfjs';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    range,
    stack,
    ones,
    scatterND,
    sub,
    mul,
} from '@tensorflow/tfjs-core';

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

// CPU fallback implementation
function efficientScatterSubCPU(args: { inputs: NamedTensorInfoMap }): TensorInfo {
    const { logits, labels, dy } = args.inputs as { logits: Tensor; labels: Tensor; dy: Tensor };
    // Use tfjs-core ops for CPU
    const batchSize = labels.shape[0];
    const depth = logits.shape[1]!;
    const batchIndices = range(0, batchSize, 1, 'int32');
    const indices = stack([batchIndices, labels], 1);
    const updates = ones([batchSize]);
    const subtractTensor = scatterND(indices, updates, [batchSize, depth]);
    const gradLogits = sub(logits, subtractTensor);
    const dyReshaped = dy.reshape([batchSize, 1]);
    const gradLogitsScaled = mul(gradLogits, dyReshaped);
    return gradLogitsScaled;
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'EfficientScatterSub',
    backendName: 'cpu',
    kernelFunc: efficientScatterSubCPU,
};

registerKernel(cpuKernelConfig);

export function scatterSub(probabilities: Tensor, labels: Tensor, scale: Tensor): Tensor {
    return engine().runKernel('EfficientScatterSub', { logits: probabilities, labels, dy: scale }, {});
}
