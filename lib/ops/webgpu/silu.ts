import { KernelConfig, NamedTensorInfoMap, registerKernel, Tensor, TensorInfo } from '@tensorflow/tfjs-core';
import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { isPackedTensor } from '@base/utilities/packed';

export class SiluProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    variableNames = ['A'];
    workgroupSize: [number, number, number];
    size = true;

    constructor(outputShape: number[]) {
        const workgroupSizeX = 128;
        this.workgroupSize = [workgroupSizeX, 1, 1];
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.shaderKey = `unary_silu`;
    }

    getUserCode(): string {
        return `
            fn unaryOperation(x : f32) -> f32 {
                return x / (1.0 + exp(-x));
            }
            ${main('index')} {
                if (index < uniforms.size) {
                let a = getAByOutputIndex(index);
                setOutputAtIndex(index, unaryOperation(a));
                }
            }
      `;
    }
}

function siluFunction(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { x } = args.inputs as { x: TensorInfo };
    const backend = args.backend as WebGPUBackend;
    const program = new SiluProgram(x.shape);
    return backend.runWebGPUProgram(program, [x], 'float32');
}

const siluConfig: KernelConfig = {
    kernelName: 'Silu',
    backendName: 'webgpu',
    kernelFunc: siluFunction,
};

registerKernel(siluConfig);

// Backward

class SiluGradProgram16 implements WebGPUProgram {
    // Inputs: dy, x
    variableNames = ['dy', 'x'];
    outputShape: number[];
    shaderKey = 'SiluGrad16';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [128, 1, 1];
    size = true;

    constructor(shape: number[]) {
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode(): string {
        return `
            fn activationGrad(dy: f32, X: f32) -> f32 {
                let sigmoid = 1.0 / (1.0 + exp(-X));
                return dy * (sigmoid + X * sigmoid * (1.0 - sigmoid));
            }
            ${main('index')} {
                if (index < uniforms.size) {
                    let X  = unpack2x16float(u32(x[index]));
                    let DY = unpack2x16float(u32(dy[index]));
                    let dsilu = vec2<f32>(
                        activationGrad(DY.x, X.x),
                        activationGrad(DY.y, X.y)
                    );
                    result[index] = i32(pack2x16float(dsilu));
                }
            }`;
    }
}

class SiluGradProgram32 implements WebGPUProgram {
    // Inputs: dy, x
    variableNames = ['dy', 'x'];
    outputShape: number[];
    shaderKey = 'SiluGrad32';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [128, 1, 1];
    size = true;

    constructor(shape: number[]) {
        this.outputShape = shape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode(): string {
        return `
            fn activationGrad(dy: f32, X: f32) -> f32 {
                let sigmoid = 1.0 / (1.0 + exp(-X));
                return dy * (sigmoid + X * sigmoid * (1.0 - sigmoid));
            }
            ${main('index')} {
                if (index < uniforms.size) {
                    let X  = getXByOutputIndex(index);
                    let DY = getDyByOutputIndex(index);
                    let dsilu = activationGrad(DY, X);
                    setOutputAtIndex(index, dsilu);
                }
            }`;
    }
}

// Backward kernel
function siluGradKernelFunc(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { dy, x } = args.inputs as { dy: Tensor; x: Tensor };
    const backend = args.backend as WebGPUBackend;
    const packed = isPackedTensor(dy);
    const program = packed ? new SiluGradProgram16(x.shape) : new SiluGradProgram32(x.shape);
    const result = backend.runWebGPUProgram(program, [dy, x], packed ? 'packedF16' : 'float32');
    return result;
}

const siluGradKernelConfig: KernelConfig = {
    kernelName: 'SiluGrad',
    backendName: 'webgpu',
    kernelFunc: siluGradKernelFunc,
};

registerKernel(siluGradKernelConfig);
