import { WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { getCoordsDataType, getCoordsXYZ } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';

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

export default class TransposeProgram16 implements WebGPUProgram {
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
