import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';

import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';

class DropoutProgram16 implements WebGPUProgram {
    variableNames = ['x'];
    outputShape: number[];
    shaderKey = 'Dropout16';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    uniforms = 'dropout: f32, seed: f32';

    constructor(shape: number[]) {
        this.shaderKey = `Dropout16`;
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode() {
        return `

        fn random(coords: vec3<i32>) -> vec2<f32> {
            let x1 = f32(coords.x * 4096 + coords.y * 256 + coords.z * 2 * 16);
            let x2 = f32(coords.x * 4096 + coords.y * 256 + (coords.z * 2 + 1) * 16);
            return vec2<f32>(fract(sin(uniforms.seed + x1) * 43758.5453123), fract(sin(uniforms.seed + x2) * 43758.5453123));
        }

        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index);
                let values = unpack2x16float(u32(x[index]));
                let keepProb = 1.0 - uniforms.dropout;
                let rand = random(coords);
                let mask = step(rand, vec2<f32>(keepProb));
                let outVal = values * mask / keepProb;
                result[index] = i32(pack2x16float(outVal));
            }
        }
        `;
    }
}

function dropoutGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { dropout, seed } = args.attrs as unknown as {
        dropout: number;
        seed: number;
    };

    const backend = args.backend as WebGPUBackend;

    const program = new DropoutProgram16(x.shape);

    const uniformData = [
        { type: 'float32', data: [dropout] },
        { type: 'float32', data: [seed] },
    ];
    const dtype = 'packedF16';
    const result = backend.runWebGPUProgram(program, [x], dtype, uniformData);
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'Dropout16',
    backendName: 'webgpu',
    kernelFunc: dropoutGPU,
};

registerKernel(kernelConfig);
