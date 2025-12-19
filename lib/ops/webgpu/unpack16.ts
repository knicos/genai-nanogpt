import { WebGPUBackend, WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';

import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';

class UnpackProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey = 'Unpack16';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    variableNames = ['x'];
    size = true;
    uniforms = 'scaling : f32,';

    constructor(outShape: number[]) {
        this.outputShape = [...outShape.slice(0, -1), outShape[outShape.length - 1] * 2];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [2, 1, 1]);
    }

    getUserCode(): string {
        return `
                ${main('index')} {
                    let outIndex = index * 2i;
                    if (outIndex < uniforms.size) {
                        let v = unpack2x16float(u32(x[index]));
                        result[outIndex] = v.x * uniforms.scaling;
                        result[outIndex + 1] = v.y * uniforms.scaling;
                    }
                }`;
    }
}

function unpackGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { scaling } = args.attrs as { scaling: number };
    const backend = args.backend as WebGPUBackend;

    const program = new UnpackProgram(x.shape);

    const uniformData = [{ type: 'float32', data: [1.0 / scaling] }];

    return backend.runWebGPUProgram(program, [x], 'float32', uniformData);
}

const kernelConfig: KernelConfig = {
    kernelName: 'Unpack16',
    backendName: 'webgpu',
    kernelFunc: unpackGPU,
};

registerKernel(kernelConfig);
