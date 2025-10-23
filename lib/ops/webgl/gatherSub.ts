import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { registerKernel, KernelConfig, TensorInfo, NamedTensorInfoMap } from '@tensorflow/tfjs-core';

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
