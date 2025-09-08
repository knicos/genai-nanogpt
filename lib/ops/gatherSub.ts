import { GPGPUProgram, MathBackendWebGL, Tensor, engine } from '@tensorflow/tfjs';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    range,
    stack,
    sub,
    gatherND,
} from '@tensorflow/tfjs-core';

class GatherSubProgram implements GPGPUProgram {
    variableNames = ['labels', 'logits', 'values'];
    outputShape: number[];
    userCode: string;

    constructor(batchSize: number) {
        this.outputShape = [batchSize];

        this.userCode = `
      void main() {
        int coords = getOutputCoords();
        int index = int(getLabelsAtOutCoords());
        float val = getValuesAtOutCoords();
        float logit = getLogits(coords, index);
        setOutput(val - logit);
      }
    `;
    }
}

function scatterSubGPU(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { logits, labels, values } = args.inputs as { logits: TensorInfo; labels: TensorInfo; values: TensorInfo };

    const backend = args.backend as MathBackendWebGL;

    const batchSize = labels.shape[0];

    const program = new GatherSubProgram(batchSize);
    return backend.runWebGLProgram(program, [labels, logits, values], 'float32');
}

const kernelConfig: KernelConfig = {
    kernelName: 'EfficientGatherSub',
    backendName: 'webgl',
    kernelFunc: scatterSubGPU,
};

registerKernel(kernelConfig);

// CPU fallback implementation
function gatherSubCPU(args: { inputs: NamedTensorInfoMap }): TensorInfo {
    const { values, labels, logits } = args.inputs as { values: Tensor; labels: Tensor; logits: Tensor };
    const batchSize = labels.shape[0];
    const batchIndices = range(0, batchSize, 1, 'int32');
    const indices = stack([batchIndices, labels], 1);
    const correctLogits = gatherND(logits, indices);

    // Cross-entropy loss: -correctLogits + logSumExp
    return sub(values, correctLogits); //.as1D();
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'EfficientGatherSub',
    backendName: 'cpu',
    kernelFunc: gatherSubCPU,
};

registerKernel(cpuKernelConfig);

export function gatherSub(values: Tensor, labels: Tensor, logits: Tensor): Tensor {
    return engine().runKernel('EfficientGatherSub', { logits, labels, values }, {});
}
