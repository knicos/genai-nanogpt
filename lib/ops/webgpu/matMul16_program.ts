import { util } from '@tensorflow/tfjs-core';
import { WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';

export default class MatMul16ProgramGeneric implements WebGPUProgram {
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
    causalMask = false;
    outputComponent?: number | undefined;
    variableComponents?: number[];
    outputIndexSnippet?: string;
    outputStrideSnippet?: string;

    constructor(batch: number, O1: number, O2: number, I1: number, I2: number, transposeA = false, transposeB = false) {
        this.transposeA = transposeA;
        this.transposeB = transposeB;
        this.variableComponents = [2, 2];
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

    private addUniform(u: string) {
        if (this.uniforms) {
            this.uniforms += `, ${u}`;
        } else {
            this.uniforms = u;
        }
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
        this.addUniform('scale: f32');
        this.scale = true;
        this.shaderKey += '_scaled';
    }

    useScaleA() {
        this.addUniform('scaleA: f32');
        this.scaleA = true;
        this.shaderKey += '_scaledA';
    }

    useScaleB() {
        this.addUniform('scaleB: f32');
        this.scaleB = true;
        this.shaderKey += '_scaledB';
    }

    useActivation(activation: 'gelu') {
        this.activation = activation;
        this.shaderKey += `_${activation}`;
    }

    useCausalMask() {
        this.causalMask = true;
        this.addUniform('pastLen: i32');
        this.shaderKey += `_causalMask`;
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
        const unpackSnippet = `
            var col = i32(localId.x);
            var row = i32(localId.y) * 4;
            var packedA: vec2<i32> = A[offsetA + row * strideA + col];
            var Arow1 = vec4<f32>(
                unpack2x16float(u32(packedA.x)),
                unpack2x16float(u32(packedA.y))
            );
            packedA = A[offsetA + (row + 1) * strideA + col];
            var Arow2 = vec4<f32>(
                unpack2x16float(u32(packedA.x)),
                unpack2x16float(u32(packedA.y))
            );
            packedA = A[offsetA + (row + 2) * strideA + col];
            var Arow3 = vec4<f32>(
                unpack2x16float(u32(packedA.x)),
                unpack2x16float(u32(packedA.y))
            );
            packedA = A[offsetA + (row + 3) * strideA + col];
            var Arow4 = vec4<f32>(
                unpack2x16float(u32(packedA.x)),
                unpack2x16float(u32(packedA.y))
            );
            
            ${this.scaleA ? 'Arow1 = Arow1 * uniforms.scaleA;' : ''}
            ${this.scaleA ? 'Arow2 = Arow2 * uniforms.scaleA;' : ''}
            ${this.scaleA ? 'Arow3 = Arow3 * uniforms.scaleA;' : ''}
            ${this.scaleA ? 'Arow4 = Arow4 * uniforms.scaleA;' : ''}
        `;

        return this.transposeA
            ? `{
                ${unpackSnippet}
                mm_Asub[row][col] = Arow1;
                mm_Asub[row + 1][col] = Arow2;
                mm_Asub[row + 2][col] = Arow3;
                mm_Asub[row + 3][col] = Arow4;
        }`
            : `{
                ${unpackSnippet}
                
                col = i32(localId.x) * 4;
                row = i32(localId.y);
                
                mm_Asub[col][row] = vec4<f32>(Arow1.x, Arow2.x, Arow3.x, Arow4.x);
                mm_Asub[col + 1][row] = vec4<f32>(Arow1.y, Arow2.y, Arow3.y, Arow4.y);
                mm_Asub[col + 2][row] = vec4<f32>(Arow1.z, Arow2.z, Arow3.z, Arow4.z);
                mm_Asub[col + 3][row] = vec4<f32>(Arow1.w, Arow2.w, Arow3.w, Arow4.w);
        }`;
    }

    /* Transpose when writing to shared memory */
    private readBSnippet(): string {
        const unpackSnippet = `
            var col = i32(localId.x);
            var row = i32(localId.y) * 4;
            var packedB: vec2<i32> = B[offsetB + row * strideB + col];
            var Brow1 = vec4<f32>(
                unpack2x16float(u32(packedB.x)),
                unpack2x16float(u32(packedB.y))
            );
            packedB = B[offsetB + (row + 1) * strideB + col];
            var Brow2 = vec4<f32>(
                unpack2x16float(u32(packedB.x)),
                unpack2x16float(u32(packedB.y))
            );
            packedB = B[offsetB + (row + 2) * strideB + col];
            var Brow3 = vec4<f32>(
                unpack2x16float(u32(packedB.x)),
                unpack2x16float(u32(packedB.y))
            );
            packedB = B[offsetB + (row + 3) * strideB + col];
            var Brow4 = vec4<f32>(
                unpack2x16float(u32(packedB.x)),
                unpack2x16float(u32(packedB.y))
            );
            
            ${this.scaleB ? 'Brow1 = Brow1 * uniforms.scaleB;' : ''}
            ${this.scaleB ? 'Brow2 = Brow2 * uniforms.scaleB;' : ''}
            ${this.scaleB ? 'Brow3 = Brow3 * uniforms.scaleB;' : ''}
            ${this.scaleB ? 'Brow4 = Brow4 * uniforms.scaleB;' : ''}
        `;

        if (this.transposeB) {
            return `{
                ${unpackSnippet}
                
                col = i32(localId.x) * 4;
                row = i32(localId.y);
                
                mm_Bsub[col][row] = vec4<f32>(Brow1.x, Brow2.x, Brow3.x, Brow4.x);
                mm_Bsub[col + 1][row] = vec4<f32>(Brow1.y, Brow2.y, Brow3.y, Brow4.y);
                mm_Bsub[col + 2][row] = vec4<f32>(Brow1.z, Brow2.z, Brow3.z, Brow4.z);
                mm_Bsub[col + 3][row] = vec4<f32>(Brow1.w, Brow2.w, Brow3.w, Brow4.w);
            }`;
        } else {
            return `{
                ${unpackSnippet}
                mm_Bsub[row][col] = Brow1;
                mm_Bsub[row + 1][col] = Brow2;
                mm_Bsub[row + 2][col] = Brow3;
                mm_Bsub[row + 3][col] = Brow4;
            }`;
        }
    }

    private baseIndexSnippets(): string {
        const strides = `
            let strideA = uniforms.aShape.z / 2;
            let strideB = uniforms.bShape.z / 2;
        `;
        let baseB = '';
        if (this.transposeB) {
            baseB = `let baseB = getIndexFromCoords3D(vec3<i32>(batchB, globalColStart, 0), vec3<i32>(uniforms.bShape.x, uniforms.bShape.y, strideB));`;
        } else {
            baseB = `let baseB = getIndexFromCoords3D(vec3<i32>(batchB, 0, globalColStart / 4), vec3<i32>(uniforms.bShape.x, uniforms.bShape.y, strideB));`;
        }

        let baseA = '';
        if (this.transposeA) {
            baseA = `let baseA = getIndexFromCoords3D(vec3<i32>(batchA, 0, globalRowStart / 4), vec3<i32>(uniforms.aShape.x, uniforms.aShape.y, strideA));`;
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
            offsetA = `let offsetA = baseA + kStart / 4;`;
        }

        let offsetB = '';
        if (this.transposeB) {
            offsetB = `let offsetB = baseB + kStart / 4;`;
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
            var<workgroup> mm_Asub : array<array<vec4<f32>, ${
                tileAWidth / 4 + (this.transposeA ? 0 : 1)
            }>, ${tileAHeight}>;
            var<workgroup> mm_Bsub : array<array<vec4<f32>, ${
                tileBOuter / 4 + (this.transposeB ? 1 : 0)
            }>, ${tileInner}>;

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

                ${this.baseIndexSnippets()}

                for (var t = 0; t < ${numTiles}; t++) {
                    ${this.offsetSnippets()}

                    ${this.readASnippet()}
                    ${this.readBSnippet()}

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

                    ${
                        this.causalMask
                            ? `
                    // Causal Masking: mask if col > row + pastLen
                    let r = gRow + i;
                    let cBase = gColPacked * 2;
                    let cVec = vec4<i32>(cBase, cBase + 1, cBase + 2, cBase + 3);
                    let mask = cVec > vec4<i32>(r + uniforms.pastLen);
                    acc[i] = select(acc[i], vec4<f32>(-uniforms.INFINITY), mask);
                    `
                            : ''
                    }

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
