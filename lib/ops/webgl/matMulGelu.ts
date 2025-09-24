import {
    KernelConfig,
    KernelFunc,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    broadcast_util,
    TensorInfo,
    upcastType,
    util,
    tidy,
    matMul,
    engine,
} from '@tensorflow/tfjs-core';
// import { CHECK_NAN_SNIPPET } from '@tensorflow/tfjs-backend-webgl/dist/unaryop_gpu';
import { MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { reshape } from '@tensorflow/tfjs-backend-webgl/dist/kernels/Reshape';
import { MatMulPackedProgram } from '@tensorflow/tfjs-backend-webgl/dist/mulmat_packed_gpu';

const K = 0.7978845608028654; // sqrt(2/pi)
const A = 0.044715;

/*const GELU =
    CHECK_NAN_SNIPPET +
    `
    float x3 = x * x * x;
    float inner = x + ${A} * x3;
    inner = ${K} * inner;
    inner = tanh(inner);
    inner = 0.5 * (1.0 + inner);
    x = x * inner;
    return x;
`;*/

const GELU_PACKED = `
    vec4 x3 = x * x * x;
    vec4 inner = x + ${A} * x3;
    inner = ${K} * inner;
    inner = tanh(inner);
    inner = 0.5 * (1.0 + inner);
    vec4 result = x * inner;
    return result;
`;

const DGELU_PACKED = `
    vec4 a2 = a * a;
    vec4 a3 = a2 * a;
    vec4 u  = ${K} * (a + ${A} * a3);
    vec4 t  = tanh(u);
    vec4 sech2 = 1.0 - t * t;
    vec4 du_dx = ${K} * (1.0 + 3.0 * ${A} * a2);
    vec4 dgelu = 0.5 * (1.0 + t) + 0.5 * a * sech2 * du_dx;
    return dgelu * b;
`;

// Empirically determined minimal shared dimension in matmul before we forward
// to a.mul(b).sum() in order to take advantage of GPU parallelism. See
// https://github.com/tensorflow/tfjs-core/pull/1379 for benchmarks.
export const MATMUL_SHARED_DIM_THRESHOLD = 1000;

type BatchMatMulConfig = {
    a: TensorInfo;
    b: TensorInfo;
    transposeA: boolean;
    transposeB: boolean;
    backend: MathBackendWebGL;
    activationSnippet?: string;
    multiplier?: TensorInfo;
};

/*
 * This is largely adapted from the batchMatMul implementation in
 * tfjs-backend-webgl, with the addition of an activation snippet to fuse GELU.
 */
export function batchMatMulGeluImpl({
    a,
    b,
    transposeA,
    transposeB,
    backend,
    activationSnippet,
    multiplier,
}: BatchMatMulConfig): TensorInfo {
    const aRank = a.shape.length;
    const bRank = b.shape.length;

    const innerShapeA = transposeA ? a.shape[aRank - 2] : a.shape[aRank - 1];
    const innerShapeB = transposeB ? b.shape[bRank - 1] : b.shape[bRank - 2];

    const outerShapeA = transposeA ? a.shape[aRank - 1] : a.shape[aRank - 2];
    const outerShapeB = transposeB ? b.shape[bRank - 2] : b.shape[bRank - 1];

    const outerDimsA = a.shape.slice(0, -2);
    const outerDimsB = b.shape.slice(0, -2);

    const batchDimA = util.sizeFromShape(outerDimsA);
    const batchDimB = util.sizeFromShape(outerDimsB);

    const outShapeOuterDims = broadcast_util.assertAndGetBroadcastShape(a.shape.slice(0, -2), b.shape.slice(0, -2));
    const outShape = outShapeOuterDims.concat([outerShapeA, outerShapeB]);

    util.assert(
        innerShapeA === innerShapeB,
        () =>
            `Error in matMul: inner shapes (${innerShapeA}) and (` +
            `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
            `${b.shape} and transposeA=${transposeA}` +
            ` and transposeB=${transposeB} must match.`
    );

    const a3dShape: [number, number, number] = transposeA
        ? [batchDimA, innerShapeA, outerShapeA]
        : [batchDimA, outerShapeA, innerShapeA];
    const b3dShape: [number, number, number] = transposeB
        ? [batchDimB, outerShapeB, innerShapeB]
        : [batchDimB, innerShapeB, outerShapeB];

    // The rest of the implementation is designed to operate on rank-3 tensors
    const a3d = reshape({ inputs: { x: a }, backend, attrs: { shape: a3dShape } });
    const b3d = reshape({ inputs: { x: b }, backend, attrs: { shape: b3dShape } });

    const intermediates: TensorInfo[] = [a3d, b3d];

    const batchDim = Math.max(batchDimA, batchDimB);

    const fusedActivation = activationSnippet;

    const dtype = upcastType(a.dtype, b.dtype);

    const program = new MatMulPackedProgram(
        a3dShape,
        b3dShape,
        [batchDim, outerShapeA, outerShapeB],
        transposeA,
        transposeB,
        false,
        fusedActivation,
        !!multiplier,
        false
    );

    const inputs: TensorInfo[] = [a3d, b3d];

    if (multiplier) {
        inputs.push(multiplier);
    }

    const out = backend.runWebGLProgram(program, inputs, dtype);

    const outReshaped = reshape({ inputs: { x: out }, backend, attrs: { shape: outShape } });
    intermediates.push(out);
    for (const i of intermediates) {
        backend.disposeIntermediateTensorInfo(i);
    }
    return outReshaped;
}

export function batchMatMulKernel(args: { inputs: { x: TensorInfo; kernel: TensorInfo }; backend: MathBackendWebGL }) {
    const { inputs, backend } = args;
    const { x, kernel } = inputs;

    if (x === undefined || kernel === undefined) {
        throw new Error('BatchMatMul requires two input tensors.');
    }

    return batchMatMulGeluImpl({
        a: x,
        b: kernel,
        transposeA: false,
        transposeB: false,
        backend,
        activationSnippet: GELU_PACKED,
    });
}

const matMulGeluConfig: KernelConfig = {
    kernelName: 'MatMulGelu',
    backendName: 'webgl',
    kernelFunc: batchMatMulKernel as unknown as KernelFunc,
};

registerKernel(matMulGeluConfig);

// Backward

// Backward kernel
function matMulGeluGradKernelFunc(args: { inputs: NamedTensorInfoMap; backend: unknown }): TensorInfo[] {
    const { dy, x, kernel } = args.inputs as { dy: Tensor; x: Tensor; kernel: Tensor };
    const backend = args.backend as MathBackendWebGL;
    //const program = new GeluGradProgram(x.shape);
    //return backend.runWebGLProgram(program, [dy, x], 'float32');

    return tidy(() => {
        // 1. Compute m = x @ kernel
        // 2. Compute dgelu/dm at m using DGELU_PACKED
        // 3. dL/dm = dy * dgelu/dm with the multiplier
        const dL_dm = engine().makeTensorFromTensorInfo(
            batchMatMulGeluImpl({
                a: x,
                b: kernel,
                transposeA: false,
                transposeB: false,
                backend,
                activationSnippet: DGELU_PACKED,
                multiplier: dy,
            })
        );

        // 4. dx = dL_dm @ kernel^T
        const dx = matMul(dL_dm, kernel, false, true);

        // 5. dkernel = x^T @ dL_dm
        const dkernel = matMul(x, dL_dm, true, false);

        return [dx, dkernel];
    });
}

const matMulGeluGradKernelConfig: KernelConfig = {
    kernelName: 'MatMulGeluGrad',
    backendName: 'webgl',
    kernelFunc: matMulGeluGradKernelFunc,
};

registerKernel(matMulGeluGradKernelConfig);
