import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';

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

class RMSNormProgram implements GPGPUProgram {
    variableNames = ['x', 'meanSquare', 'gamma'];
    outputShape: number[];
    userCode: string;

    constructor(batch: number, T: number, C: number) {
        this.outputShape = [batch, T, C];

        this.userCode = `
        void main() {
            ivec3 coords = getOutputCoords();
            float x = getXAtOutCoords();
            float meanSquare = getMeanSquare(coords.x, coords.y, 0);
            float gamma = getGammaAtOutCoords();
            float invRms = inversesqrt(meanSquare + 1e-8);
            float normalized = x * invRms;
            float outVal = normalized * gamma;
            setOutput(outVal);
        }
        `;
    }
}

function rmsNormGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x, gamma } = args.inputs as { x: Tensor; gamma: Tensor };

    const backend = args.backend as MathBackendWebGL;

    const batchSize = x.shape[0];
    const seqLength = x.shape[1]!;
    const C = x.shape[2]!;

    const meanSquare = x.square().mean(-1, true);
    const program = new RMSNormProgram(batchSize, seqLength, C);
    return backend.runWebGLProgram(program, [x, meanSquare, gamma], 'float32');
}

const kernelConfig: KernelConfig = {
    kernelName: 'RMSNorm',
    backendName: 'webgl',
    kernelFunc: rmsNormGPU,
};

registerKernel(kernelConfig);

// --- Gradient ---

class RMSNormGradXProgram implements GPGPUProgram {
    variableNames = ['x', 'meanSquare', 'dyGamma', 'dyXMean'];
    outputShape: number[];
    userCode: string;

    constructor(batch: number, T: number, C: number) {
        this.outputShape = [batch, T, C];

        this.userCode = `
        void main() {
            ivec3 coords = getOutputCoords();
            float x = getXAtOutCoords();
            float meanSquare = getMeanSquare(coords.x, coords.y, 0) + 1e-8;
            float dyGamma = getDyGammaAtOutCoords();
            float dyXMean = getDyXMean(coords.x, coords.y, 0) / ${C}.0;
            float invRms = inversesqrt(meanSquare);
            float dx = dyGamma * invRms - x * dyXMean * invRms / meanSquare;
            setOutput(dx);
        }
        `;
    }
}

class RMSNormGradGammaProgram implements GPGPUProgram {
    variableNames = ['x', 'meanSquare', 'dy'];
    outputShape: number[];
    userCode: string;

    constructor(batch: number, T: number, C: number) {
        this.outputShape = [batch, T, C];

        this.userCode = `
        void main() {
            ivec3 coords = getOutputCoords();
            float x = getXAtOutCoords();
            float meanSquare = getMeanSquare(coords.x, coords.y, 0) + 1e-8;
            float dy = getDyAtOutCoords();
            float invRms = inversesqrt(meanSquare);
            float dGamma = dy * (x * invRms);
            setOutput(dGamma);
        }
        `;
    }
}

function rmsNormGradGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo[] {
    const { dy, x, gamma } = args.inputs as { dy: Tensor; x: Tensor; gamma: Tensor };

    const backend = args.backend as MathBackendWebGL;

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
    const dx = backend.runWebGLProgram(dxProgram, [x, meanSquare, dyGamma, dyXMean], 'float32');

    dyGamma.dispose();
    dyXMean.dispose();

    const gammaProgram = new RMSNormGradGammaProgram(batchSize, seqLength, C);
    const dGammaFull = backend.runWebGLProgram(gammaProgram, [x, meanSquare, dy], 'float32');

    meanSquare.dispose();

    const dGamma = sum(engine().makeTensorFromTensorInfo(dGammaFull), [0, 1]);
    backend.disposeIntermediateTensorInfo(dGammaFull);

    return [dx, dGamma];
}

const gradKernelConfig: KernelConfig = {
    kernelName: 'RMSNormGrad',
    backendName: 'webgl',
    kernelFunc: rmsNormGradGPU,
};

registerKernel(gradKernelConfig);
