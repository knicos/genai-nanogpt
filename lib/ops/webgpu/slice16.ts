import { WebGPUBackend, WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import { getCoordsDataType, getCoordsXYZ } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import {
    KernelConfig,
    KernelFunc,
    registerKernel,
    slice_util,
    SliceAttrs,
    SliceInputs,
    TensorInfo,
    util,
} from '@tensorflow/tfjs-core';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { PackedTensorInfo } from '@base/patches/PackedTensor';

function dSnippet(rank: number): string {
    switch (rank) {
        case 1:
            return '1D';
        case 2:
            return '2D';
        case 3:
            return '3D';
        case 4:
            return '4D';
    }
    return '';
}

class SliceProgram16 implements WebGPUProgram {
    variableNames = ['source'];
    uniforms: string;
    outputShape: number[];
    shaderKey: string;
    rank: number;
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workPerThread = 1;
    workgroupSize: [number, number, number] = [64, 1, 1];
    start: number[];
    size = true;

    constructor(start: number[], destSize: number[]) {
        this.outputShape = destSize;
        this.rank = destSize.length;
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [
            this.workPerThread,
            1,
            1,
        ]);

        this.start = start;
        this.uniforms = `start : ${getCoordsDataType(start.length)}, `;
        this.shaderKey = 'slice';
    }

    getUserCode(): string {
        const dtype = getCoordsDataType(this.rank);
        let coordSum;
        if (this.start.length === 1) {
            coordSum = this.outputShape.map(() => {
                return `sourceLoc = uniforms.start + coords;`;
            });
        } else {
            coordSum = this.outputShape.map((_, i) => {
                return `sourceLoc.${coords[i]} = uniforms.start.${getCoordsXYZ(i)} + coords.${coords[i]};`;
            });
        }

        const userCode = `
      ${main('index')} {
        if (index < uniforms.size) {
          var sourceLoc : ${dtype};
          let coords = getCoordsFromIndex(index);
          ${coordSum.join('\n')}
          result[index] = source[getIndexFromCoords${dSnippet(this.rank)}(sourceLoc, uniforms.sourceShape)];
        }
      }
    `;
        return userCode;
    }
}

const coords = ['x', 'y', 'z', 'w', 'u', 'v'];

export function slice(args: { inputs: SliceInputs; backend: WebGPUBackend; attrs: SliceAttrs }): TensorInfo {
    const { inputs, backend, attrs } = args;
    const { x } = inputs;
    const { begin, size } = attrs;

    const [$begin, $size] = slice_util.parseSliceParams(x!, begin, size);
    slice_util.assertParamsValid(x!, $begin, $size);

    if (util.sizeFromShape($size) === 0) {
        return backend.makeTensorInfo($size, x!.dtype, []);
    }

    // TODO(xing.xu): Add shadow slice support.
    const program = new SliceProgram16($begin, $size);
    const uniformData = [{ type: 'int32', data: $begin }];
    const result: PackedTensorInfo = backend.runWebGPUProgram(program, [x!], x!.dtype, uniformData);
    result.packed = true;
    return result;
}

const sliceConfig: KernelConfig = {
    kernelName: 'Slice16',
    backendName: 'webgpu',
    kernelFunc: slice as unknown as KernelFunc,
};

registerKernel(sliceConfig);
