import { isPackedTensor } from '@base/utilities/packed';
import { WebGPUBackend, WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import {
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    Tensor,
    TensorInfo,
    transpose,
    TransposeAttrs,
} from '@tensorflow/tfjs-core';

import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { getCoordsDataType, getCoordsXYZ } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { PackedTensorInfo } from '@base/patches/PackedTensor';
import { unpack16 } from '../unpack16';
import { pack16 } from '../pack16';

function getSwitchedCoords(newDim: number[]): string {
    const rank = newDim.length;
    if (rank > 6) {
        throw Error(`Transpose for rank ${rank} is not yet supported`);
    }
    const switchedCoords = new Array(rank);
    for (let i = 0; i < newDim.length; i++) {
        switchedCoords[newDim[i]] = `coords.${getCoordsXYZ(i)}`;
    }

    return switchedCoords.join();
}

class TransposeProgram16 implements WebGPUProgram {
    variableNames = ['A'];
    shaderKey: string;
    outputShape: number[];
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workPerThread = 1;
    workgroupSize: [number, number, number] = [64, 1, 1];
    newDim: number[];
    size = true;

    constructor(aShape: number[], newDim: number[]) {
        const outputShape: number[] = new Array(aShape.length);
        for (let i = 0; i < outputShape.length; i++) {
            outputShape[i] = aShape[newDim[i]];
        }
        this.outputShape = outputShape;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [
            this.workPerThread,
            1,
            1,
        ]);

        this.newDim = newDim;
        this.shaderKey = `transpose16_${newDim}`;
    }

    getUserCode(): string {
        const dtype = getCoordsDataType(this.outputShape.length);
        const switched = getSwitchedCoords(this.newDim);

        const userCode = `
      ${main('index')} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            result[flatIndex] = A[getIndexFromCoords${this.outputShape.length}D(
              ${dtype}(${switched}), uniforms.aShape)];
          }
        }
      }
    `;
        return userCode;
    }
}

function transpose16_(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { inputs, attrs } = args;
    const { x } = inputs as { x: Tensor };
    const { perm } = attrs as unknown as TransposeAttrs;

    const backend = args.backend as WebGPUBackend;

    const packed = isPackedTensor(x);

    if (packed && perm[perm.length - 1] !== x.shape.length - 1) {
        // Force unpacking in this case
        const unpacked = unpack16(x);
        const transposed = transpose(unpacked, perm);
        unpacked.dispose();
        const packed = pack16(transposed);
        transposed.dispose();
        return packed;
    }

    if (packed) {
        const program = new TransposeProgram16(x.shape, perm);
        const output: PackedTensorInfo = backend.runWebGPUProgram(program, [x], 'int32');
        output.packed = true;
        return output;
    } else {
        return transpose(x, perm);
    }
}

const webgpuConfig: KernelConfig = {
    kernelName: 'Transpose16',
    backendName: 'webgpu',
    kernelFunc: transpose16_,
};
registerKernel(webgpuConfig);
