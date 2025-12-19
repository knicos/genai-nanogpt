import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
    util,
    broadcast_util,
    engine,
    matMul,
    scalar,
} from '@tensorflow/tfjs-core';
import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { isPackedTensor } from '@base/utilities/packed';
import { reshape16 } from '../reshape16';
import { matMulMul } from '../matMulMul';

type ProgramUniform = Array<{
    type: string;
    data: number[];
}>;

class MatMulAttention16ProgramGeneric implements WebGPUProgram {
    variableNames = ['A', 'B'];
    outputShape: number[];

    shaderKey = 'MatMulAttention16TB';
    dispatchLayout: { x: number[]; y: number[]; z: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [8, 8, 1]; // TILE_T2 * TILE_D = 8 * 8
    dimInner: number;
    transposeA = false;
    transposeB = true;
    broadcastBatch = true;
    tileInner = 32;
    uniforms?: string;
    scale = false;

    constructor(batch: number, O1: number, O2: number, I1: number, I2: number, transposeA = false, transposeB = false) {
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.shaderKey = `MatMulAttention16TB_${O1}_${O2}_${I1}_${I2}_${transposeA ? 'TA' : ''}${
            transposeB ? 'TB' : ''
        }`;
        if (transposeA) {
            this.outputShape = [batch, I1, I2 / 2];
            this.dimInner = O1; // or O2
            if (O1 !== O2) {
                throw new Error('Inner dimensions of A and B must match for MatMulAttention16 transposeA');
            }
        } else if (transposeB) {
            this.outputShape = [batch, O1, O2 / 2];
            this.dimInner = I2; // or I1
            if (I2 !== I1) {
                throw new Error('Inner dimensions of A and B must match for MatMulAttention16 transposeB');
            }
            // Neither transposed (both transposed is not supported)
        } else {
            this.outputShape = [batch, O1, I2 / 2];
            this.dimInner = I1; // or O2
            if (I1 !== O2) {
                throw new Error('Inner dimensions of A and B must match for MatMulAttention16');
            }
        }

        if (this.dimInner % this.tileInner !== 0) {
            throw new Error(`Inner dimension ${this.dimInner} must be multiple of ${this.tileInner}`);
        }

        this.dispatchLayout = { x: [2], y: [1], z: [0] }; //flatDispatchLayout(this.outputShape);
        // One workgroup per (b,h,t1); each WG computes a strip of t2-packed outputs.
        // WG width is 8. Each thread computes 2 packed columns (4 unpacked).
        // So we need outputShape[2] / 2 threads in X.
        this.dispatch = [
            Math.ceil(this.outputShape[2] / (this.workgroupSize[0] * 2)), // 4 unpacked cols per thread = 2 packed cols
            Math.ceil(this.outputShape[1] / (this.workgroupSize[1] * 4)), // 4 rows per thread
            this.outputShape[0],
        ];

        if (I2 % 32 !== 0) {
            throw new Error('Head size must be even for MatMulAttention16 transposeB');
        }
        if (I1 % 32 !== 0) {
            throw new Error('Head size must be even for MatMulAttention16 transposeB');
        }
        if (O1 % 32 !== 0) {
            throw new Error('Sequence length must be multiple of 32 for MatMulAttention16 transposeB');
        }
        if (O2 % 32 !== 0) {
            throw new Error('Sequence length must be multiple of 32 for MatMulAttention16 transposeB');
        }
    }

    useScale() {
        this.uniforms = 'scale: f32';
        this.scale = true;
        this.shaderKey += '_scaled';
    }

    private readASnippet(): string {
        // Workgroup 8x8 = 64 threads.
        // We need to load 32x32 floats = 1024 floats = 512 packed ints.
        // Each thread loads 512 / 64 = 8 ints.
        if (this.transposeA) {
            return `
                // tileAHeight=32, tileAWidth=32. Packed shape [32, 16].
                for (var i = 0; i < 8; i = i + 1) {
                    let localIndex = i * 64 + i32(localId.y) * 8 + i32(localId.x);
                    let row = localIndex / 16;
                    let col = localIndex % 16;
                    
                    let index = getIndexFromCoords3D(vec3<i32>(batchA, kStart + row, globalRowStart / 2 + col), uniforms.aShape);
                    let packedA = A[index];
                    let unpackedA = unpack2x16float(u32(packedA));
                    mm_Asub[row][col * 2] = unpackedA.x;
                    mm_Asub[row][col * 2 + 1] = unpackedA.y;
                }
            `;
        } else {
            return `
                // tileAHeight=32, tileAWidth=32. Packed shape [32, 16].
                for (var i = 0; i < 8; i = i + 1) {
                    let localIndex = i * 64 + i32(localId.y) * 8 + i32(localId.x);
                    let row = localIndex / 16;
                    let col = localIndex % 16;

                    let index = getIndexFromCoords3D(vec3<i32>(batchA, globalRowStart + row, kStart / 2 + col), uniforms.aShape);
                    let packedA = A[index];
                    let unpackedA = unpack2x16float(u32(packedA));
                    mm_Asub[row][col * 2] = unpackedA.x;
                    mm_Asub[row][col * 2 + 1] = unpackedA.y;
                }
            `;
        }
    }

    private readBSnippet(): string {
        // Workgroup 8x8 = 64 threads.
        // We need to load 32x32 floats = 512 packed ints.
        if (this.transposeB) {
            return `
                // tileBOuter=32, tileInner=32. We need B^T [32, 32].
                // Maps to B [32, 32]. Packed B [32, 16].
                for (var i = 0; i < 8; i = i + 1) {
                    let localIndex = i * 64 + i32(localId.y) * 8 + i32(localId.x);
                    let row = localIndex / 16;
                    let col = localIndex % 16;

                    let index = getIndexFromCoords3D(vec3<i32>(batchB, globalColStart + row, kStart / 2 + col), uniforms.bShape);
                    let packedB = B[index];
                    let unpackedB = unpack2x16float(u32(packedB));
                    // Transpose into shared memory
                    mm_Bsub[col * 2][row] = unpackedB.x;
                    mm_Bsub[col * 2 + 1][row] = unpackedB.y;
                }
            `;
        } else {
            return `
                // tileBOuter=32, tileInner=32. We need B [32, 32].
                // Packed B [32, 16].
                for (var i = 0; i < 8; i = i + 1) {
                    let localIndex = i * 64 + i32(localId.y) * 8 + i32(localId.x);
                    //let row = localIndex / 8; // Wait, 32 rows / 8? No.
                    // Packed B is [32, 16]. Total 512 ints.
                    // row should be localIndex / 16 (0..31).
                    let row = localIndex / 16;
                    let col = localIndex % 16;

                    let index = getIndexFromCoords3D(vec3<i32>(batchB, kStart + row, globalColStart / 2 + col), uniforms.bShape);
                    let packedB = B[index];
                    let unpackedB = unpack2x16float(u32(packedB));
                    mm_Bsub[row][col * 2] = unpackedB.x;
                    mm_Bsub[row][col * 2 + 1] = unpackedB.y;
                }
            `;
        }
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

        // ...assertions...

        const userCode = `
            var<workgroup> mm_Asub : array<array<f32, ${tileAWidth + 1}>, ${tileAHeight}>;
            var<workgroup> mm_Bsub : array<array<f32, ${tileBOuter + 1}>, ${tileInner}>;

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

                for (var t = 0; t < ${numTiles}; t++) {
                    ${this.readASnippet()}
                    ${this.readBSnippet()}
                    kStart = kStart + ${tileInner};
                    workgroupBarrier();

                    let lCol4 = localCol * 4;
                    let lRow4 = localRow * 4;

                    for (var k = 0; k < ${tileInner}; k++) {
                        // Load 4 columns of B as a vec4
                        let bVec = vec4<f32>(
                            mm_Bsub[k][lCol4],
                            mm_Bsub[k][lCol4 + 1],
                            mm_Bsub[k][lCol4 + 2],
                            mm_Bsub[k][lCol4 + 3]
                        );

                        // Compute 4 rows
                        let a0 = ${transposeA ? `mm_Asub[k][lRow4]` : `mm_Asub[lRow4][k]`};
                        acc[0] = fma(vec4<f32>(a0), bVec, acc[0]);

                        let a1 = ${transposeA ? `mm_Asub[k][lRow4 + 1]` : `mm_Asub[lRow4 + 1][k]`};
                        acc[1] = fma(vec4<f32>(a1), bVec, acc[1]);

                        let a2 = ${transposeA ? `mm_Asub[k][lRow4 + 2]` : `mm_Asub[lRow4 + 2][k]`};
                        acc[2] = fma(vec4<f32>(a2), bVec, acc[2]);

                        let a3 = ${transposeA ? `mm_Asub[k][lRow4 + 3]` : `mm_Asub[lRow4 + 3][k]`};
                        acc[3] = fma(vec4<f32>(a3), bVec, acc[3]);
                    }
                    workgroupBarrier();
                }

                // Write out 4 rows x 2 packed cols (4 unpacked cols)
                let gRow = globalRowStart + localRow * 4;
                let gColPacked = i32(workgroupId.x) * ${this.workgroupSize[0] * 2} + localCol * 2;
                
                // Row 0
                let idx0 = getOutputIndexFromCoords(vec3<i32>(batch, gRow, gColPacked));
                ${this.scale ? `acc[0] = acc[0] * uniforms.scale;` : ''}
                result[idx0] = i32(pack2x16float(acc[0].xy));
                result[idx0 + 1] = i32(pack2x16float(acc[0].zw));

                // Row 1
                let idx1 = idx0 + uniforms.outShapeStrides[1];
                ${this.scale ? `acc[1] = acc[1] * uniforms.scale;` : ''}
                result[idx1] = i32(pack2x16float(acc[1].xy));
                result[idx1 + 1] = i32(pack2x16float(acc[1].zw));

                // Row 2
                let idx2 = idx1 + uniforms.outShapeStrides[1];
                ${this.scale ? `acc[2] = acc[2] * uniforms.scale;` : ''}
                result[idx2] = i32(pack2x16float(acc[2].xy));
                result[idx2 + 1] = i32(pack2x16float(acc[2].zw));

                // Row 3
                let idx3 = idx2 + uniforms.outShapeStrides[1];
                ${this.scale ? `acc[3] = acc[3] * uniforms.scale;` : ''}
                result[idx3] = i32(pack2x16float(acc[3].xy));
                result[idx3 + 1] = i32(pack2x16float(acc[3].zw));
            }
        `;
        return userCode;
    }
}

function matMul16GPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { A, B } = args.inputs as { A: Tensor; B: Tensor };
    const { transposeA, transposeB, scale } = args.attrs as {
        transposeA: boolean;
        transposeB: boolean;
        scale?: number;
    };

    const backend = args.backend as WebGPUBackend;

    const wasAUnpacked = !isPackedTensor(A);
    const wasBUnpacked = !isPackedTensor(B);

    if (wasAUnpacked && wasBUnpacked) {
        if (scale !== undefined) {
            return matMulMul(A, B, scalar(scale), transposeA, transposeB);
        }
        return matMul(A, B, transposeA, transposeB);
    }

    const aRank = A.shape.length;
    const bRank = B.shape.length;

    const outerDimsA = A.shape.slice(0, -2);
    const outerDimsB = B.shape.slice(0, -2);

    const batchDimA = util.sizeFromShape(outerDimsA);
    const batchDimB = util.sizeFromShape(outerDimsB);

    const outShapeOuterDims = broadcast_util.assertAndGetBroadcastShape(A.shape.slice(0, -2), B.shape.slice(0, -2));

    const batchSize = Math.max(batchDimA, batchDimB);
    const O1 = A.shape[aRank - 2]!; // Sequence length
    const O2 = B.shape[bRank - 2]!; // Sequence length
    const I1 = A.shape[aRank - 1]! * 2; // Head size, * 2 because of packing
    const I2 = B.shape[bRank - 1]! * 2; // Head size, * 2 because of packing
    //assertShapesMatch(B.shape, [batchSize, nh, T2, hs / 2], 'Error in MatMulAttention16: ');

    const Areshaped = reshape16(A, [batchDimA, A.shape[aRank - 2]!, A.shape[aRank - 1]!]);
    const BReshaped = reshape16(B, [batchDimB, B.shape[bRank - 2]!, B.shape[bRank - 1]!]);

    const program = new MatMulAttention16ProgramGeneric(batchSize, O1, O2, I1, I2, transposeA, transposeB);

    let uniforms: ProgramUniform | undefined;

    if (scale !== undefined) {
        program.useScale();
        uniforms = [{ type: 'float32', data: [scale] }];
    }

    const result = backend.runWebGPUProgram(program, [Areshaped, BReshaped], 'int32', uniforms);
    Areshaped.dispose();
    BReshaped.dispose();

    const resultRank = result.shape.length;
    const outShape = outShapeOuterDims.concat([result.shape[resultRank - 2], result.shape[resultRank - 1]]);

    const resultTensor = engine().makeTensorFromTensorInfo(result);
    const resultReshaped = reshape16(resultTensor, outShape);
    resultTensor.dispose();
    return resultReshaped;
}

const kernelConfig: KernelConfig = {
    kernelName: 'MatMul16',
    backendName: 'webgpu',
    kernelFunc: matMul16GPU,
};

registerKernel(kernelConfig);
