import type { WebGPUBackend, WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
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

class PackProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey = 'Pack16';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    variableNames = ['x'];
    uniforms = 'scaling : f32,';
    size = true;

    constructor(outShape: number[]) {
        this.outputShape = [...outShape.slice(0, -1), Math.ceil(outShape[outShape.length - 1] / 2)];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode(): string {
        return `
                ${main('index')} {
                    if (index < uniforms.size) {
                    let inputIndex = index * 2;
                    let v = vec2<f32>(x[inputIndex] * uniforms.scaling, x[inputIndex + 1] * uniforms.scaling);
                    let packed = pack2x16float(v);
                    result[index] = i32(packed);
                    }
                }`;
    }
}

function packGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { scaling } = args.attrs as { scaling: number };
    const backend = args.backend as WebGPUBackend;

    const program = new PackProgram(x.shape);

    const uniformData = [{ type: 'float32', data: [scaling] }];

    return backend.runWebGPUProgram(program, [x], 'int32', uniformData);
}

const kernelConfig: KernelConfig = {
    kernelName: 'Pack16',
    backendName: 'webgpu',
    kernelFunc: packGPU,
};

registerKernel(kernelConfig);
