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
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';

class AppendCacheProgram implements WebGPUProgram {
    variableNames = ['cache', 'item'];
    outputShape: number[];
    shaderKey = 'AppendCache';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    uniforms = 'cacheT: i32';

    constructor(batch: number, nh: number, T: number, hs: number, maxSize: number) {
        const outT = Math.min(T + 1, maxSize);
        this.shaderKey = `AppendCache_${outT}`;
        this.outputShape = [batch, nh, outT, hs];
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);
    }

    getUserCode() {
        const maxSize = this.outputShape[2];
        return `
        ${main('index')} {
            if (index < uniforms.size) {
                let coords = getCoordsFromIndex(index); // [b, h, t, d]
                let b = coords[0];
                let h = coords[1];
                let t = coords[2];
                let d = coords[3];

                let itemT = 1;
                let maxSize = ${maxSize};
                let totalT = uniforms.cacheT + itemT;
                let start = select(0, 1, totalT >= maxSize);

                let srcT = t + start;
                var val = 0.0;
                if (srcT < uniforms.cacheT) {
                    val = getCache(b, h, srcT, d);
                }
                if (srcT == uniforms.cacheT) {
                    val = getItem(b, h, 0, d);
                }

                setOutputAtIndex(index, val);
            }
        }
        `;
    }
}

function appendCacheGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { cache, item } = args.inputs as { cache: Tensor; item: Tensor };
    const { maxSize, pastLen } = args.attrs as { maxSize: number; pastLen: number };

    const backend = args.backend as WebGPUBackend;

    const batchSize = cache.shape[0];
    const T = cache.shape[2]!; // Sequence length
    const nh = cache.shape[1]!; // Number of heads

    assertShapesMatch(item.shape, [batchSize, nh, 1, item.shape[3]!], 'Error in AppendCache: ');

    if (pastLen < 0 || pastLen > maxSize) {
        throw new Error(`Invalid pastLen value: ${pastLen}. Must be in the range [0, ${maxSize}].`);
    }

    const program = new AppendCacheProgram(batchSize, nh, T, item.shape[3]!, maxSize);
    const uniformData = [{ type: 'int32', data: [pastLen] }];
    return backend.runWebGPUProgram(program, [cache, item], 'float32', uniformData);
}

const kernelConfig: KernelConfig = {
    kernelName: 'AppendCache',
    backendName: 'webgpu',
    kernelFunc: appendCacheGPU,
};

registerKernel(kernelConfig);
