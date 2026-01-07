import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
    util,
    broadcast_util,
    matMul,
    scalar,
    reshape,
    transpose,
    mul,
} from '@tensorflow/tfjs-core';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { isPackedTensor } from '@base/utilities/packed';
import { reshape16 } from '../reshape16';
import { matMulMul } from '../matMulMul';
import { matMulGelu } from '../matMulGelu';
import MatMul16ProgramGeneric from './matMul16_program';

type ProgramUniform = Array<{
    type: string;
    data: number[];
}>;

function matMul16GPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { A, B } = args.inputs as { A: Tensor; B: Tensor };
    const { transposeA, transposeB, scale, activation, scaleA, scaleB, forceOutputShape, perm, causalMask, pastLen } =
        args.attrs as {
            transposeA: boolean;
            transposeB: boolean;
            scale?: number;
            scaleA?: number;
            scaleB?: number;
            activation?: 'gelu';
            forceOutputShape?: number[];
            perm?: number[];
            originalShape?: number[];
            causalMask?: boolean;
            pastLen?: number;
        };

    const backend = args.backend as WebGPUBackend;

    const wasAUnpacked = !isPackedTensor(A);
    const wasBUnpacked = !isPackedTensor(B);

    // Unpacked 32-bit float version uses standard matMul
    // This also adds the optional features but without fusion
    if (wasAUnpacked && wasBUnpacked) {
        const sA = scaleA !== undefined ? mul(A, scalar(scaleA)) : A;
        const sB = scaleB !== undefined ? mul(B, scalar(scaleB)) : B;

        // TODO: Causal mask.
        if (causalMask) {
            throw new Error('Causal mask is not supported for unpacked MatMul16.');
        }

        let result: Tensor;
        if (scale !== undefined) {
            result = matMulMul(sA, sB, scalar(scale), transposeA, transposeB);
        } else if (activation === 'gelu') {
            result = matMulGelu(sA, sB);
        } else {
            result = matMul(sA, sB, transposeA, transposeB);
        }

        // Apply forced output shape and perm if needed
        if (perm) {
            if (forceOutputShape) {
                const reshaped = reshape(result, forceOutputShape);
                result.dispose();
                const permuted = transpose(reshaped, perm);
                reshaped.dispose();
                return permuted;
            } else {
                const permuted = transpose(result, perm);
                result.dispose();
                return permuted;
            }
        } else if (forceOutputShape) {
            const r = reshape(result, forceOutputShape);
            result.dispose();
            return r;
        } else {
            return result;
        }
    }

    if (wasAUnpacked && !wasBUnpacked) {
        throw new Error('When using mixed precision, A must be packed if B is packed.');
    }

    if (!wasAUnpacked && wasBUnpacked) {
        throw new Error('When using mixed precision, B must be packed if A is packed.');
    }

    const aRank = A.shape.length;
    const bRank = B.shape.length;

    const outerDimsA = A.shape.slice(0, -2);
    const outerDimsB = B.shape.slice(0, -2);

    const batchDimA = util.sizeFromShape(outerDimsA);
    const batchDimB = util.sizeFromShape(outerDimsB);

    const outShapeOuterDims = broadcast_util.assertAndGetBroadcastShape(A.shape.slice(0, -2), B.shape.slice(0, -2));

    const batchSize = Math.max(batchDimA, batchDimB);
    const O1 = A.shape[aRank - 2]!;
    const O2 = B.shape[bRank - 2]!;
    const I1 = A.shape[aRank - 1]! * 2; // *2 because of packing
    const I2 = B.shape[bRank - 1]! * 2; // *2 because of packing

    const Areshaped = reshape16(A, [batchDimA, A.shape[aRank - 2]!, A.shape[aRank - 1]!]);
    const BReshaped = reshape16(B, [batchDimB, B.shape[bRank - 2]!, B.shape[bRank - 1]!]);

    const program = new MatMul16ProgramGeneric(batchSize, O1, O2, I1, I2, transposeA, transposeB);

    const uniforms: ProgramUniform = [];

    // Output scale
    if (scale !== undefined) {
        program.useScale();
        uniforms.push({ type: 'float32', data: [scale] });
    }
    if (scaleA !== undefined) {
        program.useScaleA();
        uniforms.push({ type: 'float32', data: [scaleA] });
    }
    if (scaleB !== undefined) {
        program.useScaleB();
        uniforms.push({ type: 'float32', data: [scaleB] });
    }

    if (activation !== undefined) {
        program.useActivation(activation);
    }

    if (causalMask) {
        program.useCausalMask();
        uniforms.push({ type: 'int32', data: [pastLen || 0] });
    }

    const resultRank = program.outputShape.length;
    if (forceOutputShape) {
        args.attrs!.originalShape = program.outputShape;
    }

    const outShape =
        forceOutputShape ??
        outShapeOuterDims.concat([program.outputShape[resultRank - 2], program.outputShape[resultRank - 1]]);
    program.setOutputShape(outShape, perm);

    const result = backend.runWebGPUProgram(
        program,
        [Areshaped, BReshaped],
        'packedF16',
        uniforms.length > 0 ? uniforms : undefined
    );
    Areshaped.dispose();
    BReshaped.dispose();
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'MatMul16',
    backendName: 'webgpu',
    kernelFunc: matMul16GPU,
};

registerKernel(kernelConfig);
