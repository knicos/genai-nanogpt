import { KernelConfig, NamedTensorInfoMap, registerKernel, Tensor, TensorInfo } from '@tensorflow/tfjs-core';
import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';

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
      fn polyTanh(x: f32) -> f32 {
         return select(tanh(x), sign(x), abs(x) > 15.0);
      }
      fn unaryOperation(x : f32) -> f32 {
        let x3 = x * x * x;
        var inner = fma(${A}, x3, x);
        inner = ${K} * inner;
        inner = polyTanh(inner);
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

class GeluGradProgram implements WebGPUProgram {
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
        // d/dx gelu(x) = 0.5*(1 + t) + 0.5*x*(1 - t^2)*k*(1 + 3a x^2)
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode(): string {
        return `
            fn polyTanh(x: f32) -> f32 {
                return select(tanh(x), sign(x), abs(x) > 15.0);
            }
            ${main('index')} {
                if (index < uniforms.size) {
                    let X  = getXByOutputIndex(index);
                    let x2 = X * X;
                    let x3 = x2 * X;
                    let u  = ${K} * (X + ${A} * x3);
                    let t  = polyTanh(u);
                    let sech2 = 1.0 - t * t;
                    let du_dx = ${K} * (1.0 + 3.0 * ${A} * x2);
                    let dgelu = 0.5 * (1.0 + t) + 0.5 * X * sech2 * du_dx;
                    let DY = getDyByOutputIndex(index);
                    setOutputAtIndex(index, DY * dgelu);
                }
            }`;
    }
}

// Backward kernel
function geluGradKernelFunc(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo {
    const { dy, x } = args.inputs as { dy: Tensor; x: Tensor };
    const backend = args.backend as WebGPUBackend;
    const program = new GeluGradProgram(x.shape);
    return backend.runWebGPUProgram(program, [dy, x], 'float32');
}

const geluGradKernelConfig: KernelConfig = {
    kernelName: 'GeluGrad',
    backendName: 'webgpu',
    kernelFunc: geluGradKernelFunc,
};

registerKernel(geluGradKernelConfig);
