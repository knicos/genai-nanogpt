import { Tensor, engine } from '@tensorflow/tfjs-core';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    concat,
} from '@tensorflow/tfjs-core';

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
