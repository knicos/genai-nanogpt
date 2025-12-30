import { computeDispatch } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { getMainHeaderString as main, WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { util } from '@tensorflow/tfjs-core';

export default class TransposeSharedProgram16 implements WebGPUProgram {
    variableNames = ['A'];
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: { x: number[]; y: number[]; z?: number[] };
    dispatch: [number, number, number];
    // Note that the maximum number of workgroup invocations by webgpu is 256.
    // Nick: Reduce to 8x8
    workgroupSize: [number, number, number] = [8, 8, 1];

    constructor(aShape: number[], newDim: number[]) {
        const rank = aShape.length;
        const outputShape: number[] = new Array(rank);
        const aCorrectedShape = aShape.slice();
        aCorrectedShape[aCorrectedShape.length - 1] *= 2; // Because inner dim is packed
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aCorrectedShape[newDim[i]];
        }
        outputShape[outputShape.length - 1] /= 2; // Because inner dim is packed
        this.outputShape = outputShape;
        this.dispatchLayout = rank === 2 ? { x: [0], y: [1] } : { x: [1], y: [2], z: [0] };
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [2, 1, 1]);

        this.shaderKey = `transposeShared16_${rank}`;
    }

    getUserCode(): string {
        const rank = this.outputShape.length;
        util.assert(
            this.workgroupSize[0] === this.workgroupSize[1],
            () => `Must be a square tile, current tile shape is ${this.workgroupSize[0]} x ${this.workgroupSize[1]}`
        );
        const tileSize = this.workgroupSize[0] * 2;
        const userCode = `
      var<workgroup> tile : array<array<f32, ${tileSize + 1}>, ${tileSize}>;
      ${main()} {
        var x = i32(workgroupId.x) * ${tileSize / 2} + i32(localId.x);
        var y = i32(workgroupId.y) * ${tileSize} + i32(localId.y);
        let batch = ${rank === 3 ? 'i32(workgroupId.z)' : '0'};
        let batchOffsetA = ${rank === 3 ? 'batch * uniforms.aShapeStrides[0]' : '0'};
        let batchOffsetOut = ${rank === 3 ? 'batch * uniforms.outShapeStrides[0]' : '0'};

        let inputWidth = uniforms.outShape[${rank === 3 ? '1' : '0'}] / 2; // Output height
        let inputHeight = uniforms.outShape[${rank === 3 ? '2' : '1'}] * 2; // Output width
        if (x < inputWidth && y < inputHeight) {
            let unpackedA = unpack2x16float(u32(A[batchOffsetA + y * inputWidth + x]));
            tile[localId.y][localId.x * 2] = unpackedA.x;
            tile[localId.y][localId.x * 2 + 1] = unpackedA.y;
        }
        // Second load to cover the tile
        y = y + ${this.workgroupSize[0]};
        if (x < inputWidth && y < inputHeight) {
            let unpackedA = unpack2x16float(u32(A[batchOffsetA + y * inputWidth + x]));
            tile[localId.y + ${this.workgroupSize[0]}][localId.x * 2] = unpackedA.x;
            tile[localId.y + ${this.workgroupSize[0]}][localId.x * 2 + 1] = unpackedA.y;
        }
        workgroupBarrier();

        let outputWidth = uniforms.outShape[${rank === 3 ? '2' : '1'}]; // Output width
        let outputHeight = uniforms.outShape[${rank === 3 ? '1' : '0'}] * 2; // Output height
        x = i32(workgroupId.y) * ${tileSize / 2} + i32(localId.x);
        y = i32(workgroupId.x) * ${tileSize} + i32(localId.y);
        if (x < outputWidth && y < outputHeight) {
          result[batchOffsetOut + y * outputWidth + x] = i32(pack2x16float(vec2<f32>(tile[localId.x * 2][localId.y], tile[localId.x * 2 + 1][localId.y])));
        }
        // Second store to cover the tile
        y = y + ${this.workgroupSize[0]};
        if (x < outputWidth && y < outputHeight) {
          result[batchOffsetOut + y * outputWidth + x] = i32(pack2x16float(vec2<f32>(tile[localId.x * 2][localId.y + ${
              this.workgroupSize[0]
          }], tile[localId.x * 2 + 1][localId.y + ${this.workgroupSize[0]}])));
        }
      }
    `;
        return userCode;
    }
}
