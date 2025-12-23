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
    uniforms?: string;
    outputComponent = 4;
    variableComponents = [2];
    scaling = false;

    constructor(outShape: number[]) {
        this.outputShape = [...outShape.slice(0, -1), outShape[outShape.length - 1] * 2];
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
                    let outIndex = index;
                    if (outIndex < uniforms.size) {
                        let xvec2 = x[index];
                        let v1 = vec4<f32>(
                            unpack2x16float(u32(xvec2.x)),
                            unpack2x16float(u32(xvec2.y))
                        ) ${this.scaling ? '* uniforms.scaling' : ''};
                        result[outIndex] = v1;
                    }
                }`;
    }
}

function unpackGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { scaling } = args.attrs as { scaling: number };
    const backend = args.backend as WebGPUBackend;

    const program = new UnpackProgram(x.shape);

    const hasScaling = scaling !== 1.0;
    if (hasScaling) {
        program.useScaling();
    }

    const uniformData = [{ type: 'float32', data: [1.0 / scaling] }];

    return backend.runWebGPUProgram(program, [x], 'float32', hasScaling ? uniformData : undefined);
}

const kernelConfig: KernelConfig = {
    kernelName: 'Unpack16',
    backendName: 'webgpu',
    kernelFunc: unpackGPU,
};

registerKernel(kernelConfig);
