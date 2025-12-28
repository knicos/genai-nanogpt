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
import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { isPackedTensor } from '@base/utilities/packed';
import { reshape16 } from '../reshape16';
import { matMulMul } from '../matMulMul';
import { matMulGelu } from '../matMulGelu';
import { PackedTensorInfo } from '@base/patches/PackedTensor';

type ProgramUniform = Array<{
    type: string;
    data: number[];
}>;

class MatMul16ProgramGeneric implements WebGPUProgram {
    variableNames = ['A', 'B'];
    outputShape: number[];

    shaderKey = 'MatMul16TB';
    dispatchLayout: { x: number[]; y: number[]; z: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [8, 8, 1]; // 8x8 threads for 32x32 tile
    dimInner: number;
    transposeA = false;
    transposeB = true;
    broadcastBatch = true;
    tileInner = 32;
    uniforms?: string;
    scale = false;
    scaleA = false;
    scaleB = false;
    activation?: 'gelu';
    outputComponent?: number | undefined;
    variableComponents?: number[];
    outputIndexSnippet?: string;
    outputStrideSnippet?: string;

    constructor(batch: number, O1: number, O2: number, I1: number, I2: number, transposeA = false, transposeB = false) {
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.variableComponents = [4, 4];
        this.outputComponent = 2;

        this.shaderKey = `MatMul16TB_${O1}_${O2}_${I1}_${I2}_${transposeA ? 'TA' : ''}${transposeB ? 'TB' : ''}`;
        if (transposeA) {
            this.outputShape = [batch, I1, I2 / 2];
            this.dimInner = O1; // or O2
            if (O1 !== O2) {
                throw new Error('Inner dimensions of A and B must match for MatMul16 transposeA');
            }
        } else if (transposeB) {
            this.outputShape = [batch, O1, O2 / 2];
            this.dimInner = I2; // or I1
            if (I2 !== I1) {
                throw new Error('Inner dimensions of A and B must match for MatMul16 transposeB');
            }
            // Neither transposed (both transposed is not supported)
        } else {
            this.outputShape = [batch, O1, I2 / 2];
            this.dimInner = I1; // or O2
            if (I1 !== O2) {
                throw new Error('Inner dimensions of A and B must match for MatMul16');
            }
        }

        if (this.dimInner % this.tileInner !== 0) {
            throw new Error(`Inner dimension ${this.dimInner} must be multiple of ${this.tileInner}`);
        }

        this.dispatchLayout = { x: [2], y: [1], z: [0] };

        this.dispatch = [
            Math.ceil(this.outputShape[2] / (this.workgroupSize[0] * 2)), // 4 unpacked cols per thread = 2 packed cols
            Math.ceil(this.outputShape[1] / (this.workgroupSize[1] * 4)), // 4 rows per thread
            this.outputShape[0],
        ];

        if (I2 % 32 !== 0) {
            throw new Error('Head size must be even for MatMul16 transposeB');
        }
        if (I1 % 32 !== 0) {
            throw new Error('Head size must be even for MatMul16 transposeB');
        }
        if (O1 % 32 !== 0) {
            throw new Error('Sequence length must be multiple of 32 for MatMul16 transposeB');
        }
        if (O2 % 32 !== 0) {
            throw new Error('Sequence length must be multiple of 32 for MatMul16 transposeB');
        }

        this.outputIndexSnippet = `var idx0 = getOutputIndexFromCoords(vec3<i32>(batch, gRow, gColPacked));`;
        this.outputStrideSnippet = `idx0 = idx0 + uniforms.outShapeStrides[1];  // Next row`;
    }

    /* Note: this is done after constructor because it shouldn't affect dispatch */
    setOutputShape(shape: number[], perm?: number[]) {
        const newSize = util.sizeFromShape(shape);
        const currentSize = util.sizeFromShape(this.outputShape);
        if (newSize !== currentSize) {
            throw new Error(`New shape size ${newSize} must match current size ${currentSize}`);
        }

        // Helper to split an index into two dims
        function splitIndex(idx: string, dim2: number): string[] {
            return [`${idx} / ${dim2}`, `${idx} % ${dim2}`];
        }

        // Detect if batch or last dim was split
        const oldShape = this.outputShape;
        let logicalIndices: string[] = [];

        // Note, this is always 4 since rank max is 4.
        if (shape.length === oldShape.length + 1) {
            // One dim was split into two
            // Check batch split
            if (shape[0] * shape[1] === oldShape[0]) {
                // Batch split: [B, ...] -> [B1, B2, ...]
                logicalIndices = [
                    ...splitIndex('batch', shape[1]), // batch / B2, batch % B2
                    'gRow',
                    'gColPacked',
                ];
                this.shaderKey += `_batchSplit_${shape[1]}`;
            } else if (shape[shape.length - 2] * shape[shape.length - 1] === oldShape[oldShape.length - 1]) {
                // Last dim split: [..., N] -> [..., N1, N2]
                logicalIndices = [
                    'batch',
                    'gRow',
                    ...splitIndex('gColPacked', shape[shape.length - 1]), // gColPacked / N2, gColPacked % N2
                ];
                this.shaderKey += `_colSplit_${shape[shape.length - 1]}`;
            } else {
                throw new Error('Unsupported output shape split');
            }
        } else if (shape.length === oldShape.length) {
            // No split, just reshape or transpose
            logicalIndices = ['batch', 'gRow', 'gColPacked'];
        } else if (shape.length === 2 && oldShape[0] === 1) {
            // Batch dim was removed
            logicalIndices = ['gRow', 'gColPacked'];
            this.shaderKey += `_batchRemoved`;
        } else {
            throw new Error(`Unsupported output shape rank change: ${oldShape.length} -> ${shape.length}}`);
        }

        let coordExprs: string[] = [];
        if (perm) {
            if (perm.length !== shape.length) {
                throw new Error('Permutation length must match output rank');
            }
            coordExprs = perm.map((i) => logicalIndices[i]);
            this.shaderKey += `_perm_${perm.join('')}`;
        } else {
            coordExprs = logicalIndices;
        }

        const strideIdx = coordExprs.findIndex((expr) => expr === 'gRow');

        const coordVec = `vec${shape.length}<i32>(${coordExprs.join(', ')})`;

        // These provide the indexing and stride for correct output permutation and shape
        this.outputIndexSnippet = `var idx0: i32 = getOutputIndexFromCoords(${coordVec});`;
        this.outputStrideSnippet = `idx0 = idx0 + uniforms.outShapeStrides${strideIdx === 0 ? '' : `[${strideIdx}]`}; `;

        // Finally, set the correct output shape for the WebGPU data buffer
        if (perm) {
            this.outputShape = perm.map((i) => shape[i]);
        } else {
            this.outputShape = shape;
        }
    }

    useScale() {
        this.uniforms = 'scale: f32';
        this.scale = true;
        this.shaderKey += '_scaled';
    }

    useScaleA() {
        this.uniforms = 'scaleA: f32';
        this.scaleA = true;
        this.shaderKey += '_scaledA';
    }

    useScaleB() {
        this.uniforms = 'scaleB: f32';
        this.scaleB = true;
        this.shaderKey += '_scaledB';
    }

    useActivation(activation: 'gelu') {
        this.activation = activation;
        this.shaderKey += `_${activation}`;
    }

    private activationSnippet(): string {
        if (this.activation === 'gelu') {
            const K = 0.7978845608028654; // sqrt(2/pi)
            const A = 0.044715;
            return `
                // TODO: revisit after https://github.com/gpuweb/gpuweb/issues/4458 is resolved
                fn tanhComplete(x: vec4<f32>) -> vec4<f32> {
                    return vec4<f32>(
                        select(tanh(x.x), sign(x.x), abs(x.x) > 15.0f),
                        select(tanh(x.y), sign(x.y), abs(x.y) > 15.0f),
                        select(tanh(x.z), sign(x.z), abs(x.z) > 15.0f),
                        select(tanh(x.w), sign(x.w), abs(x.w) > 15.0f),
                    );
                }
                fn activation(x : vec4<f32>) -> vec4<f32> {
                    let x3 = x * x * x;
                    var inner = fma(vec4<f32>(${A}f), x3, x);
                    inner = ${K}f * inner;
                    inner = tanhComplete(inner);
                    inner = 0.5f * (1.0f + inner);
                    return x * inner;
                }
                `;
        }
        return '';
    }

    /* Transpose when writing to shared memory */
    private readASnippet(): string {
        const loadIndex = `let indexA = offsetA + row * strideA + col;`;

        const unpackSnippet = `
            let packedA = A[indexA];
            var unpackedA1 = vec4<f32>(
                unpack2x16float(u32(packedA.x)),
                unpack2x16float(u32(packedA.y))
            );
            var unpackedA2 = vec4<f32>(
                unpack2x16float(u32(packedA.z)),
                unpack2x16float(u32(packedA.w))
            );
            ${this.scaleA ? 'unpackedA1 = unpackedA1 * uniforms.scaleA;' : ''}
            ${this.scaleA ? 'unpackedA2 = unpackedA2 * uniforms.scaleA;' : ''}
        `;

        if (this.transposeA) {
            return `{
                ${loadIndex}
                ${unpackSnippet}
                mm_Asub[row][col * 2] = unpackedA1;
                mm_Asub[row][col * 2 + 1] = unpackedA2;
        }`;
        } else {
            return `{
                ${loadIndex}
                ${unpackSnippet}
                let cx = row / 4;
                let cy = row % 4;
                let colBase = col * 8;
                mm_Asub[colBase][cx][cy] = unpackedA1.x;
                mm_Asub[colBase + 1][cx][cy] = unpackedA1.y;
                mm_Asub[colBase + 2][cx][cy] = unpackedA1.z;
                mm_Asub[colBase + 3][cx][cy] = unpackedA1.w;
                mm_Asub[colBase + 4][cx][cy] = unpackedA2.x;
                mm_Asub[colBase + 5][cx][cy] = unpackedA2.y;
                mm_Asub[colBase + 6][cx][cy] = unpackedA2.z;
                mm_Asub[colBase + 7][cx][cy] = unpackedA2.w;
        }`;
        }
    }

    /* Transpose when writing to shared memory */
    private readBSnippet(): string {
        const loadIndex = `let indexB = offsetB + row * strideB + col;`;

        const unpackSnippet = `
            let packedB = B[indexB];
            var unpackedB1 = vec4<f32>(
                unpack2x16float(u32(packedB.x)),
                unpack2x16float(u32(packedB.y))
            );
            var unpackedB2 = vec4<f32>(
                unpack2x16float(u32(packedB.z)),
                unpack2x16float(u32(packedB.w))
            );
            ${this.scaleB ? 'unpackedB1 = unpackedB1 * uniforms.scaleB;' : ''}
            ${this.scaleB ? 'unpackedB2 = unpackedB2 * uniforms.scaleB;' : ''}
        `;

        if (this.transposeB) {
            return `{
                ${loadIndex}
                ${unpackSnippet}
                // Transpose into shared memory, reorganise the vec4s
                let rx = row / 4;
                let ry = row % 4;
                let colBase = col * 8;
                mm_Bsub[colBase][rx][ry] = unpackedB1.x;
                mm_Bsub[colBase + 1][rx][ry] = unpackedB1.y;
                mm_Bsub[colBase + 2][rx][ry] = unpackedB1.z;
                mm_Bsub[colBase + 3][rx][ry] = unpackedB1.w;
                mm_Bsub[colBase + 4][rx][ry] = unpackedB2.x;
                mm_Bsub[colBase + 5][rx][ry] = unpackedB2.y;
                mm_Bsub[colBase + 6][rx][ry] = unpackedB2.z;
                mm_Bsub[colBase + 7][rx][ry] = unpackedB2.w;
            }`;
        } else {
            return `{
                ${loadIndex}
                ${unpackSnippet}
                mm_Bsub[row][col * 2] = unpackedB1;
                mm_Bsub[row][col * 2 + 1] = unpackedB2;
            }`;
        }
    }

    private baseIndexSnippets(): string {
        const strides = `
            let strideA = uniforms.aShape.z / 4;
            let strideB = uniforms.bShape.z / 4;
        `;
        let baseB = '';
        if (this.transposeB) {
            baseB = `let baseB = getIndexFromCoords3D(vec3<i32>(batchB, globalColStart, 0), vec3<i32>(uniforms.bShape.x, uniforms.bShape.y, strideB));`;
        } else {
            baseB = `let baseB = getIndexFromCoords3D(vec3<i32>(batchB, 0, globalColStart / 8), vec3<i32>(uniforms.bShape.x, uniforms.bShape.y, strideB));`;
        }

        let baseA = '';
        if (this.transposeA) {
            baseA = `let baseA = getIndexFromCoords3D(vec3<i32>(batchA, 0, globalRowStart / 8), vec3<i32>(uniforms.aShape.x, uniforms.aShape.y, strideA));`;
        } else {
            baseA = `let baseA = getIndexFromCoords3D(vec3<i32>(batchA, globalRowStart, 0), vec3<i32>(uniforms.aShape.x, uniforms.aShape.y, strideA));`;
        }

        return `
            ${strides}
            ${baseA}
            ${baseB}
        `;
    }

    private offsetSnippets(): string {
        let offsetA = '';
        if (this.transposeA) {
            offsetA = `let offsetA = baseA + kStart * strideA;`;
        } else {
            offsetA = `let offsetA = baseA + kStart / 8;`;
        }

        let offsetB = '';
        if (this.transposeB) {
            offsetB = `let offsetB = baseB + kStart / 8;`;
        } else {
            offsetB = `let offsetB = baseB + kStart * strideB;`;
        }

        return `
            ${offsetA}
            ${offsetB}
        `;
    }

    getUserCode(): string {
        const transposeA = this.transposeA;
        const tileInner = this.tileInner;
        // 4x4 tiling per thread -> 32x32 tile per WG
        const tileAOuter = this.workgroupSize[1] * 4; // 32
        const tileBOuter = this.workgroupSize[0] * 4; // 32
        const tileAWidth = transposeA ? tileAOuter : tileInner;
        const tileAHeight = transposeA ? tileInner : tileAOuter;
        const dimInner = this.dimInner;
        const numTiles = Math.ceil(dimInner / tileInner);

        const userCode = `
            var<workgroup> mm_Asub : array<array<vec4<f32>, ${tileAWidth / 4}>, ${tileAHeight}>;
            var<workgroup> mm_Bsub : array<array<vec4<f32>, ${tileBOuter / 4}>, ${tileInner}>;

            ${this.activation ? this.activationSnippet() : ''}

            ${main()} {
                let batch = i32(globalId.z);
                let batchA = ${!this.broadcastBatch ? 'batch' : 'batch % uniforms.aShape[0]'};
                let batchB = ${!this.broadcastBatch ? 'batch' : 'batch % uniforms.bShape[0]'};
                var kStart = 0;
                let localRow = i32(localId.y);
                let localCol = i32(localId.x);
                let globalRowStart = i32(workgroupId.y) * ${tileAOuter};
                let globalColStart = i32(workgroupId.x) * ${tileBOuter};

                // 4 rows x 4 cols accumulator
                // acc[i] holds row i (4 cols)
                var acc = array<vec4<f32>, 4>(
                    vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)
                );

                let offset = i32(localId.y) * ${this.workgroupSize[0]} + i32(localId.x);

                ${this.baseIndexSnippets()}

                for (var t = 0; t < ${numTiles}; t++) {
                    ${this.offsetSnippets()}

                    for (var i = 0; i < ${2 * 64}; i = i + 64) {
                        let localIndex = i + offset;
                        let row = localIndex / 4;
                        let col = localIndex % 4;
                        ${this.readASnippet()}
                        ${this.readBSnippet()}
                    }
                    kStart = kStart + ${tileInner};
                    workgroupBarrier();

                    for (var k = 0; k < ${tileInner}; k++) {
                        // Load 4 columns of B as a vec4
                        let bVec = mm_Bsub[k][localCol];
                        let aVec = mm_Asub[k][localRow];

                        // Compute 4 rows
                        for (var r = 0; r < 4; r = r + 1) {
                            acc[r] = fma(vec4<f32>(aVec[r]), bVec, acc[r]);
                        }
                    }
                    workgroupBarrier();
                }

                // Write out 4 rows x 2 packed cols (4 unpacked cols)
                let gRow = globalRowStart + localRow * 4;
                let gColPacked = i32(workgroupId.x) * ${this.workgroupSize[0] * 2} + localCol * 2;
                
                ${this.outputIndexSnippet}
                for (var i = 0; i < 4; i = i + 1) {
                    ${this.scale ? `acc[i] = acc[i] * uniforms.scale;` : ''}
                    ${this.activation ? `acc[i] = activation(acc[i]);` : ''}
                    result[idx0 / 2] = vec2<i32>(
                        i32(pack2x16float(acc[i].xy)),
                        i32(pack2x16float(acc[i].zw))
                    );

                    ${this.outputStrideSnippet}
                }
            }
        `;
        return userCode;
    }
}

function matMul16GPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { A, B } = args.inputs as { A: Tensor; B: Tensor };
    const { transposeA, transposeB, scale, activation, scaleA, scaleB, forceOutputShape, perm } = args.attrs as {
        transposeA: boolean;
        transposeB: boolean;
        scale?: number;
        scaleA?: number;
        scaleB?: number;
        activation?: 'gelu';
        forceOutputShape?: number[];
        perm?: number[];
        originalShape?: number[];
    };

    const backend = args.backend as WebGPUBackend;

    const wasAUnpacked = !isPackedTensor(A);
    const wasBUnpacked = !isPackedTensor(B);

    // Unpacked 32-bit float version uses standard matMul
    // This also adds the optional features but without fusion
    if (wasAUnpacked && wasBUnpacked) {
        const sA = scaleA !== undefined ? mul(A, scalar(scaleA)) : A;
        const sB = scaleB !== undefined ? mul(B, scalar(scaleB)) : B;

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

    let uniforms: ProgramUniform | undefined;

    // Output scale
    if (scale !== undefined) {
        program.useScale();
        uniforms = [{ type: 'float32', data: [scale] }];
    }
    if (scaleA !== undefined) {
        program.useScaleA();
        uniforms = [{ type: 'float32', data: [scaleA] }];
    }
    if (scaleB !== undefined) {
        program.useScaleB();
        uniforms = [{ type: 'float32', data: [scaleB] }];
    }

    if (activation !== undefined) {
        program.useActivation(activation);
    }

    const resultRank = program.outputShape.length;
    if (forceOutputShape) {
        args.attrs!.originalShape = program.outputShape;
    }

    const outShape =
        forceOutputShape ??
        outShapeOuterDims.concat([program.outputShape[resultRank - 2], program.outputShape[resultRank - 1]]);
    program.setOutputShape(outShape, perm);

    const result: PackedTensorInfo = backend.runWebGPUProgram(program, [Areshaped, BReshaped], 'int32', uniforms);
    result.packed = true;
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
