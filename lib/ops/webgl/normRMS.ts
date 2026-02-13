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
    variableNames = ['x', 'meanSquare'];
    outputShape: number[];
    userCode: string;

    constructor(batch: number, T: number, C: number, hasGamma = true) {
        if (hasGamma) {
            this.variableNames.push('gamma');
        }
        this.outputShape = [batch, T, C];

        this.userCode = `
        void main() {
            ivec3 coords = getOutputCoords();
            float x = getXAtOutCoords();
            float meanSquare = getMeanSquare(coords.x, coords.y, 0);
            ${hasGamma ? 'float gamma = getGammaAtOutCoords();' : ''}
            float invRms = inversesqrt(meanSquare + 1e-8);
            float normalized = x * invRms;
            float outVal = normalized ${hasGamma ? ' * gamma' : ''};
            setOutput(outVal);
        }
        `;
    }
}

function rmsNormGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x, gamma } = args.inputs as { x: Tensor; gamma?: Tensor };

    const backend = args.backend as MathBackendWebGL;

    const batchSize = x.shape[0];
    const seqLength = x.shape[1]!;
    const C = x.shape[2]!;

    const meanSquare = x.square().mean(-1, true);
    const program = new RMSNormProgram(batchSize, seqLength, C, gamma !== undefined);
    return backend.runWebGLProgram(program, gamma ? [x, meanSquare, gamma] : [x, meanSquare], 'float32');
}

const kernelConfig: KernelConfig = {
    kernelName: 'RMSNorm',
    backendName: 'webgl',
    kernelFunc: rmsNormGPU,
};

const kernelConfigNoGamma: KernelConfig = {
    kernelName: 'RMSNormNoGamma',
    backendName: 'webgl',
    kernelFunc: rmsNormGPU,
};

registerKernel(kernelConfig);
registerKernel(kernelConfigNoGamma);

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
    const { dy, x, gamma } = args.inputs as { dy: Tensor; x: Tensor; gamma?: Tensor };

    const backend = args.backend as MathBackendWebGL;

    const batchSize = x.shape[0];
    const seqLength = x.shape[1]!;
    const C = x.shape[2]!;

    const dyGamma = gamma ? dy.mul(gamma) : dy;
    const dyGammaX = dyGamma.mul(x);
    const dyXMean = dyGammaX.sum(-1, true);
    dyGammaX.dispose();

    const x2 = x.square();
    const meanSquare = x2.mean(-1, true);
    x2.dispose();

    const dxProgram = new RMSNormGradXProgram(batchSize, seqLength, C);
    const dx = backend.runWebGLProgram(dxProgram, [x, meanSquare, dyGamma, dyXMean], 'float32');

    if (gamma) {
        dyGamma.dispose();
    }
    dyXMean.dispose();

    if (gamma) {
        const gammaProgram = new RMSNormGradGammaProgram(batchSize, seqLength, C);
        const dGammaFull = backend.runWebGLProgram(gammaProgram, [x, meanSquare, dy], 'float32');

        meanSquare.dispose();

        const dGamma = sum(engine().makeTensorFromTensorInfo(dGammaFull), [0, 1]);
        backend.disposeIntermediateTensorInfo(dGammaFull);

        return [dx, dGamma];
    } else {
        meanSquare.dispose();
        return [dx];
    }
}

const gradKernelConfig: KernelConfig = {
    kernelName: 'RMSNormGrad',
    backendName: 'webgl',
    kernelFunc: rmsNormGradGPU,
};

registerKernel(gradKernelConfig);
