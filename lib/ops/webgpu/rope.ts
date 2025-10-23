import { WebGPUProgram, WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';

import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';

class RopeProgram implements WebGPUProgram {
    variableNames = ['x', 'sin', 'cos'];
    outputShape: number[];
    shaderKey = 'Rope';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    uniforms = 'pastLen: i32';

    constructor(batch: number, heads: number, T: number, C: number) {
        this.shaderKey = `Rope_${C}`;
        this.outputShape = [batch, heads, T, C];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode() {
        const C = this.outputShape[3];
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index); // [b, h, t, d]
                let b = coords[0];
                let h = coords[1];
                let t = coords[2];
                let d = coords[3];

                let rotaryDim = ${C};

                var outVal = 0.0;

                if (d < rotaryDim) {
                    let pairIdx = d / 2;
                    let cos = getCos(t + uniforms.pastLen, pairIdx, 0);
                    let sin = getSin(t + uniforms.pastLen, pairIdx, 0);

                    let ownX = getX(b, h, t, d) * cos;

                    if (d % 2 == 0) {
                        // even index
                        let even = ownX;
                        let odd = getX(b, h, t, d + 1);
                        outVal = even - odd * sin;
                    } else {
                        // odd index
                        let even = getX(b, h, t, d - 1);
                        let odd = ownX;
                        outVal = even * sin + odd;
                    }
                } else {
                    // pass through for non-rotary dims
                    outVal = getX(b, h, t, d);
                }

                setOutputAtIndex(index, outVal);
            }
        }
        `;
    }
}

function ropeGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x, sin, cos } = args.inputs as { x: Tensor; sin: Tensor; cos: Tensor };
    const { pastLen } = args.attrs as { pastLen: number };

    const backend = args.backend as WebGPUBackend;

    const batchSize = x.shape[0];
    const heads = x.shape[1]!;
    const seqLength = x.shape[2]!;
    const C = x.shape[3]!;

    const program = new RopeProgram(batchSize, heads, seqLength, C);
    const uniformData = [{ type: 'int32', data: [pastLen] }];
    return backend.runWebGPUProgram(program, [x, sin, cos], 'float32', uniformData);
}

const kernelConfig: KernelConfig = {
    kernelName: 'Rope',
    backendName: 'webgpu',
    kernelFunc: ropeGPU,
};

registerKernel(kernelConfig);
