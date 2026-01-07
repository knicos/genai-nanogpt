import { KernelConfig, NamedTensorInfoMap, registerKernel, Tensor, TensorInfo } from '@tensorflow/tfjs-core';
import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { isPackedTensor } from '@base/utilities/packed';

const K = 0.7978845608028654; // sqrt(2/pi)
const A = 0.044715;

export class GeluProgram implements WebGPUProgram {
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
        this.shaderKey = `unary_gelu`;
    }

    getUserCode(): string {
        return `
            // TODO: revisit after https://github.com/gpuweb/gpuweb/issues/4458 is resolved
            fn tanhComplete(x: f32) -> f32 {
                return select(tanh(x), sign(x), abs(x) > 15.0);
            }
            fn unaryOperation(x : f32) -> f32 {
                let x3 = x * x * x;
                var inner = fma(${A}, x3, x);
                inner = ${K} * inner;
                inner = tanhComplete(inner);
                inner = 0.5 * (1.0 + inner);
                return x * inner;
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

function geluFunction(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { x } = args.inputs as { x: TensorInfo };
    const backend = args.backend as WebGPUBackend;
    const program = new GeluProgram(x.shape);
    return backend.runWebGPUProgram(program, [x], 'float32');
}

const geluConfig: KernelConfig = {
    kernelName: 'Gelu',
    backendName: 'webgpu',
    kernelFunc: geluFunction,
};

registerKernel(geluConfig);

// Backward

class GeluGradProgram16 implements WebGPUProgram {
    // Inputs: dy, x
    variableNames = ['dy', 'x'];
    outputShape: number[];
    shaderKey = 'GeluGrad';
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
            // TODO: revisit after https://github.com/gpuweb/gpuweb/issues/4458 is resolved
            fn tanhComplete(x: f32) -> f32 {
                return select(tanh(x), sign(x), abs(x) > 15.0);
            }
            fn activationGrad(dy: f32, X: f32) -> f32 {
                let x2 = X * X;
                let x3 = x2 * X;
                let u  = ${K} * (X + ${A} * x3);
                let t  = tanhComplete(u);
                let sech2 = 1.0 - t * t;
                let du_dx = ${K} * (1.0 + 3.0 * ${A} * x2);
                let dgelu = 0.5 * (1.0 + t) + 0.5 * X * sech2 * du_dx;
                return dy *dgelu;
            }
            ${main('index')} {
                if (index < uniforms.size) {
                    let X  = unpack2x16float(u32(x[index]));
                    let DY = unpack2x16float(u32(dy[index]));
                    let dgelu = vec2<f32>(
                        activationGrad(DY.x, X.x),
                        activationGrad(DY.y, X.y)
                    );
                    result[index] = i32(pack2x16float(dgelu));
                }
            }`;
    }
}

class GeluGradProgram32 implements WebGPUProgram {
    // Inputs: dy, x
    variableNames = ['dy', 'x'];
    outputShape: number[];
    shaderKey = 'GeluGrad';
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
            // TODO: revisit after https://github.com/gpuweb/gpuweb/issues/4458 is resolved
            fn tanhComplete(x: f32) -> f32 {
                return select(tanh(x), sign(x), abs(x) > 15.0);
            }
            fn activationGrad(dy: f32, X: f32) -> f32 {
                let x2 = X * X;
                let x3 = x2 * X;
                let u  = ${K} * (X + ${A} * x3);
                let t  = tanhComplete(u);
                let sech2 = 1.0 - t * t;
                let du_dx = ${K} * (1.0 + 3.0 * ${A} * x2);
                let dgelu = 0.5 * (1.0 + t) + 0.5 * X * sech2 * du_dx;
                return dy *dgelu;
            }
            ${main('index')} {
                if (index < uniforms.size) {
                    let X  = getXByOutputIndex(index);
                    let DY = getDyByOutputIndex(index);
                    let dgelu = activationGrad(DY, X);
                    setOutputAtIndex(index, dgelu);
                }
            }`;
    }
}

// Backward kernel
function geluGradKernelFunc(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { dy, x } = args.inputs as { dy: Tensor; x: Tensor };
    const backend = args.backend as WebGPUBackend;
    const packed = isPackedTensor(dy);
    const program = packed ? new GeluGradProgram16(x.shape) : new GeluGradProgram32(x.shape);
    const result = backend.runWebGPUProgram(program, [dy, x], packed ? 'packedF16' : 'float32');
    return result;
}

const geluGradKernelConfig: KernelConfig = {
    kernelName: 'GeluGrad',
    backendName: 'webgpu',
    kernelFunc: geluGradKernelFunc,
};

registerKernel(geluGradKernelConfig);
