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
    uniforms?: string;
    size = true;
    outputComponent = 4;
    scaling = false;

    constructor(outShape: number[]) {
        this.outputShape = [...outShape.slice(0, -1), Math.ceil(outShape[outShape.length - 1] / 2)];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [4, 1, 1]);
    }

    useScaling() {
        this.shaderKey += '_Scaled';
        this.uniforms = 'scaling : f32,';
        this.scaling = true;
    }

    getUserCode(): string {
        return `
                ${main('index')} {
                    if (index < uniforms.size) {
                        let baseInputIndex =  index * 2;
                        let x1 = x[baseInputIndex] ${this.scaling ? '* uniforms.scaling' : ''};
                        let x2 = x[baseInputIndex + 1] ${this.scaling ? '* uniforms.scaling' : ''};
                        let packed = vec4<i32>(
                            i32(pack2x16float(vec2<f32>(x1.x, x1.y))),
                            i32(pack2x16float(vec2<f32>(x1.z, x1.w))),
                            i32(pack2x16float(vec2<f32>(x2.x, x2.y))),
                            i32(pack2x16float(vec2<f32>(x2.z, x2.w)))
                        );
                        result[index] = packed;
                    }
                }`;
    }
}

function packGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { scaling } = args.attrs as { scaling: number };
    const backend = args.backend as WebGPUBackend;

    const program = new PackProgram(x.shape);

    const hasScaling = scaling !== 1.0;
    if (hasScaling) {
        program.useScaling();
    }

    const uniformData = [{ type: 'float32', data: [scaling] }];

    return backend.runWebGPUProgram(program, [x], 'int32', hasScaling ? uniformData : undefined);
}

const kernelConfig: KernelConfig = {
    kernelName: 'Pack16',
    backendName: 'webgpu',
    kernelFunc: packGPU,
};

registerKernel(kernelConfig);
