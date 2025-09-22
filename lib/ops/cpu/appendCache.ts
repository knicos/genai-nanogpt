import { Tensor } from '@tensorflow/tfjs-core';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    concat,
} from '@tensorflow/tfjs-core';

// Always returns [B, H, maxSize, HS], inserts item at pastLen, shifts if full
function appendCacheCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo {
    const { cache, item } = args.inputs as { cache: Tensor; item: Tensor };
    const { maxSize, pastLen } = args.attrs as { maxSize: number; pastLen: number };

    const B = cache.shape[0]!;
    const H = cache.shape[1]!;
    const HS = cache.shape[3]!;
    const Titem = item.shape[2]!; // usually 1

    // If not full, just insert at pastLen
    if (pastLen + Titem <= maxSize) {
        const before = cache.slice([0, 0, 0, 0], [B, H, pastLen, HS]);
        const after = cache.slice([0, 0, pastLen + Titem, 0], [B, H, maxSize - pastLen - Titem, HS]);
        const itemToInsert = Titem < Titem ? item.slice([0, 0, 0, 0], [B, H, Titem, HS]) : item;
        const newCache = concat([before, itemToInsert, after], 2);
        before.dispose();
        after.dispose();
        if (itemToInsert !== item) itemToInsert.dispose();
        return newCache;
    }

    // If full, shift left and insert at the end
    // Drop the first Titem tokens, append item at the end
    const shifted = cache.slice([0, 0, Titem, 0], [B, H, maxSize - Titem, HS]);
    const newCache = concat([shifted, item], 2);
    shifted.dispose();
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
