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
    sum,
    engine,
} from '@tensorflow/tfjs-core';

class RMSNormProgram implements WebGPUProgram {
    variableNames = ['x', 'meanSquare', 'gamma'];
    outputShape: number[];

    shaderKey = 'RMSNorm';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;

    constructor(batch: number, T: number, C: number) {
        this.outputShape = [batch, T, C];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode() {
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index);
                let x = getXByOutputIndex(index);
                let meanSquare = getMeanSquare(coords[0], coords[1], 0);
                let gamma = getGammaByOutputIndex(index);
                let invRms = inverseSqrt(meanSquare + 1e-8);
                let normalized = x * invRms;
                let outVal = normalized * gamma;
                setOutputAtIndex(index, outVal);
            }
        }
        `;
    }
}

function rmsNormGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x, gamma } = args.inputs as { x: Tensor; gamma: Tensor };

    const backend = args.backend as WebGPUBackend;

    const batchSize = x.shape[0];
    const seqLength = x.shape[1]!;
    const C = x.shape[2]!;

    const meanSquare = x.square().mean(-1, true);
    const program = new RMSNormProgram(batchSize, seqLength, C);
    return backend.runWebGPUProgram(program, [x, meanSquare, gamma], 'float32');
}

const kernelConfig: KernelConfig = {
    kernelName: 'RMSNorm',
    backendName: 'webgpu',
    kernelFunc: rmsNormGPU,
};

registerKernel(kernelConfig);

// --- Gradient ---

class RMSNormGradXProgram implements WebGPUProgram {
    variableNames = ['x', 'meanSquare', 'dyGamma', 'dyXMean'];
    outputShape: number[];
    shaderKey = 'RMSNormGradX';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    C: number;

    constructor(batch: number, T: number, C: number) {
        this.outputShape = [batch, T, C];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
        this.C = C;
    }

    getUserCode() {
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index);
                let x = getXByOutputIndex(index);
                let meanSquare = getMeanSquare(coords[0], coords[1], 0) + 1e-8;
                let dyGamma = getDyGammaByOutputIndex(index);
                let dyXMean = getDyXMean(coords[0], coords[1], 0) / ${this.C}.0;
                let invRms = inverseSqrt(meanSquare);
                let dx = dyGamma * invRms - x * dyXMean * invRms / meanSquare;
                setOutputAtIndex(index, dx);
            }
        }
        `;
    }
}

class RMSNormGradGammaProgram implements WebGPUProgram {
    variableNames = ['x', 'meanSquare', 'dy'];
    outputShape: number[];
    shaderKey = 'RMSNormGradGamma';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;

    constructor(batch: number, T: number, C: number) {
        this.outputShape = [batch, T, C];

        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode() {
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index);
                let x = getXByOutputIndex(index);
                let meanSquare = getMeanSquare(coords[0], coords[1], 0) + 1e-8;
                let dy = getDyByOutputIndex(index);
                let invRms = inverseSqrt(meanSquare);
                let dGamma = dy * (x * invRms);
                setOutputAtIndex(index,dGamma);
            }
        }
        `;
    }
}

function rmsNormGradGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo[] {
    const { dy, x, gamma } = args.inputs as { dy: Tensor; x: Tensor; gamma: Tensor };

    const backend = args.backend as WebGPUBackend;

    const batchSize = x.shape[0];
    const seqLength = x.shape[1]!;
    const C = x.shape[2]!;

    const dyGamma = dy.mul(gamma);
    const dyGammaX = dyGamma.mul(x);
    const dyXMean = dyGammaX.sum(-1, true);
    dyGammaX.dispose();

    const x2 = x.square();
    const meanSquare = x2.mean(-1, true);
    x2.dispose();

    const dxProgram = new RMSNormGradXProgram(batchSize, seqLength, C);
    const dx = backend.runWebGPUProgram(dxProgram, [x, meanSquare, dyGamma, dyXMean], 'float32');

    dyGamma.dispose();
    dyXMean.dispose();

    const gammaProgram = new RMSNormGradGammaProgram(batchSize, seqLength, C);
    const dGammaFull = backend.runWebGPUProgram(gammaProgram, [x, meanSquare, dy], 'float32');

    meanSquare.dispose();

    const dGamma = sum(engine().makeTensorFromTensorInfo(dGammaFull), [0, 1]);
    backend.disposeData(dGammaFull);

    return [dx, dGamma];
}

const gradKernelConfig: KernelConfig = {
    kernelName: 'RMSNormGrad',
    backendName: 'webgpu',
    kernelFunc: rmsNormGradGPU,
};

registerKernel(gradKernelConfig);
