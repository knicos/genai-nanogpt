import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { registerKernel, KernelConfig, TensorInfo, NamedTensorInfoMap } from '@tensorflow/tfjs-core';
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';

class SoftmaxSubOneHotProgram implements WebGPUProgram {
    variableNames = ['labels', 'softmaxProbs', 'dy'];
    outputShape: number[];
    shaderKey = 'ScatterSub';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;

    constructor(batchSize: number, depth: number) {
        this.outputShape = [batchSize, depth];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode() {
        return `
            ${main('index')} {
                if (index < uniforms.size) {
                    let coords = getCoordsFromIndex(index); // [batch, depth]
                    let idx = i32(labels[coords[0]]);
                    let prob = softmaxProbs[index];
                    let dy = dy[coords[0]];
                    setOutputAtIndex(index, select(prob, prob - 1.0, idx == coords[1]) * dy);
                }
            }
            `;
    }
}

function efficientScatterSub(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { logits, labels, dy } = args.inputs as { logits: TensorInfo; labels: TensorInfo; dy: TensorInfo };

    const backend = args.backend as WebGPUBackend;

    const batchSize = labels.shape[0];
    const depth = logits.shape[1];

    assertShapesMatch(dy.shape, [batchSize], 'Error in EfficientScatterSub dy: ');
    assertShapesMatch(logits.shape, [batchSize, depth], 'Error in EfficientScatterSub logits: ');
    assertShapesMatch(labels.shape, [batchSize], 'Error in EfficientScatterSub labels: ');

    const program = new SoftmaxSubOneHotProgram(batchSize, depth);
    return backend.runWebGPUProgram(program, [labels, logits, dy], 'float32');
}

const kernelConfig: KernelConfig = {
    kernelName: 'EfficientScatterSub',
    backendName: 'webgpu',
    kernelFunc: efficientScatterSub,
};

registerKernel(kernelConfig);
