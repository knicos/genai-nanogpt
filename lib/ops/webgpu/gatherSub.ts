import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { registerKernel, KernelConfig, TensorInfo, NamedTensorInfoMap } from '@tensorflow/tfjs-core';

class GatherSubProgram implements WebGPUProgram {
    variableNames = ['labels', 'logits', 'values'];
    outputShape: number[];
    shaderKey = 'GatherSub';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;

    constructor(batchSize: number) {
        this.outputShape = [batchSize];

        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }
    getUserCode() {
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index);
                let idx = i32(getLabelsByOutputIndex(index));
                let val = getValuesByOutputIndex(index);

                if (idx < uniforms.logitsShape[1] && idx >= 0) {
                    let logit = getLogits(coords, idx);
                    setOutputAtIndex(index, val - logit);
                } else {
                    setOutputAtIndex(index, val);
                }
            }
        }
    `;
    }
}

function scatterSubGPU(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { logits, labels, values } = args.inputs as { logits: TensorInfo; labels: TensorInfo; values: TensorInfo };

    const backend = args.backend as WebGPUBackend;

    const batchSize = labels.shape[0];

    const program = new GatherSubProgram(batchSize);
    return backend.runWebGPUProgram(program, [labels, logits, values], 'float32');
}

const kernelConfig: KernelConfig = {
    kernelName: 'EfficientGatherSub',
    backendName: 'webgpu',
    kernelFunc: scatterSubGPU,
};

registerKernel(kernelConfig);
