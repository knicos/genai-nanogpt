import { GPGPUProgram, MathBackendWebGL, Tensor, engine } from '@tensorflow/tfjs';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    concat,
} from '@tensorflow/tfjs-core';

class AppendCacheProgram implements GPGPUProgram {
    variableNames = ['cache', 'item'];
    outputShape: number[];
    userCode: string;
    // enableShapeUniforms = true;
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
            } else {
                val = getItem(b, h, 0, d);
            }
            setOutput(val);
        }
        `;
    }
}

function appendCacheGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { cache, item } = args.inputs as { cache: Tensor; item: Tensor };
    const { maxSize } = args.attrs as { maxSize: number };

    const backend = args.backend as MathBackendWebGL;

    const batchSize = cache.shape[0];
    const T = cache.shape[2]!; // Sequence length
    const nh = cache.shape[1]!; // Number of heads

    const program = new AppendCacheProgram(batchSize, nh, T, item.shape[3]!, maxSize);
    return backend.runWebGLProgram(program, [cache, item], 'float32', [[T]]);
}

const kernelConfig: KernelConfig = {
    kernelName: 'AppendCache',
    backendName: 'webgl',
    kernelFunc: appendCacheGPU,
};

registerKernel(kernelConfig);

// CPU fallback implementation
function appendCacheCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo {
    const { cache, item } = args.inputs as { cache: Tensor; item: Tensor };
    const { maxSize } = args.attrs as { maxSize: number };

    const newCache = concat([cache, item], 2); // [B,nh,T_total,hs]
    const Ttotal = newCache.shape[2]!;
    if (Ttotal > maxSize) {
        const start = Ttotal - maxSize;
        const B = newCache.shape[0]!;
        const H = newCache.shape[1]!;
        const HS = newCache.shape[3]!;
        const sliced = newCache.slice([0, 0, start, 0], [B, H, maxSize, HS]);
        newCache.dispose();
        return sliced;
    }
    return newCache;
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'AppendCache',
    backendName: 'cpu',
    kernelFunc: appendCacheCPU,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'AppendCache',
    backendName: 'tensorflow',
    kernelFunc: appendCacheCPU,
};

registerKernel(tensorflowKernelConfig);

export function appendCache(cache: Tensor, item: Tensor, maxSize: number): Tensor {
    return engine().runKernel('AppendCache', { cache, item }, { maxSize });
}

/*const appendCacheGradConfig: GradConfig = {
    kernelName: 'AppendCache',
    inputsToSave: ['cache'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        if (Array.isArray(dy)) {
            throw new Error('Expected single tensor input.');
        }
        const cache = saved[0];
        const T = cache.shape[2]!; // original sequence length

        // dy: [B, nh, outT, hs], outT = min(T+1, maxSize)
        // cache: [B, nh, T, hs]
        // item: [B, nh, 1, hs]

        // Gradient for cache: first T elements along axis 2
        const dCache = dy.slice([0, 0, 0, 0], [dy.shape[0], dy.shape[1]!, T, dy.shape[3]!]);
        // Gradient for item: last element along axis 2
        const dItem = dy.slice([0, 0, dy.shape[2]! - 1, 0], [dy.shape[0], dy.shape[1]!, 1, dy.shape[3]!]);

        return { cache: () => dCache, item: () => dItem };
    },
};

registerGradient(appendCacheGradConfig);*/
