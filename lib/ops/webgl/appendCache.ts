import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';

class AppendCacheProgram implements GPGPUProgram {
    variableNames = ['cache', 'item'];
    outputShape: number[];
    userCode: string;
    customUniforms = [{ name: 'cacheT', type: 'int' as UniformType }];

    constructor(batch: number, nh: number, T: number, hs: number, maxSize: number) {
        const outT = Math.min(T + 1, maxSize);
        this.outputShape = [batch, nh, outT, hs];

        this.userCode = `
        void main() {
            ivec4 coords = getOutputCoords(); // [b, h, t, d]
            int b = coords.x;
            int h = coords.y;
            int t = coords.z;
            int d = coords.w;

            int itemT = 1;
            int maxSize = ${maxSize};
            int totalT = cacheT + itemT;
            int start = totalT >= maxSize ? 1 : 0;

            int srcT = t + start;
            float val = 0.0;
            if (srcT < cacheT) {
                val = getCache(b, h, srcT, d);
            } else if (srcT == cacheT) {
                val = getItem(b, h, 0, d);
            } else {
                val = 0.0;}
            setOutput(val);
        }
        `;
    }
}

function appendCacheGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { cache, item } = args.inputs as { cache: Tensor; item: Tensor };
    const { maxSize, pastLen } = args.attrs as { maxSize: number; pastLen: number };

    const backend = args.backend as MathBackendWebGL;

    const batchSize = cache.shape[0];
    const T = cache.shape[2]!; // Sequence length
    const nh = cache.shape[1]!; // Number of heads

    const program = new AppendCacheProgram(batchSize, nh, T, item.shape[3]!, maxSize);
    return backend.runWebGLProgram(program, [cache, item], 'float32', [[pastLen]]);
}

const kernelConfig: KernelConfig = {
    kernelName: 'AppendCache',
    backendName: 'webgl',
    kernelFunc: appendCacheGPU,
};

registerKernel(kernelConfig);
