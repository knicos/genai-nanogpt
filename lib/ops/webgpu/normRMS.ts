import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';
import { createReduceInfo, reduce } from './utils/reductions';
import { assertShapesMatch } from '@tensorflow/tfjs-core/dist/util_base';
import { isPackedTensor } from '@base/utilities/packed';
import { pack16 } from '../pack16';
import RMSProgram16 from './normRMS16_program';
import RMSProgram32 from './normRMS32_program';
import WebGPUBackendPatch from '@base/patches/webgpu_backend';
import createDeviceInformation from './utils/deviceInfo';

function rmsNormGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x, gamma } = args.inputs as { x: Tensor; gamma: Tensor };
    const backend = args.backend as WebGPUBackendPatch;

    const deviceInfo = createDeviceInformation(backend);

    const packedX = isPackedTensor(x);
    const packedGamma = isPackedTensor(gamma);
    const packed = packedX || packedGamma;

    const pX = !packed || packedX ? x : pack16(x);
    const pGamma = !packed || packedGamma ? gamma : pack16(gamma);

    const inputs = [pX, pGamma];
    const reduceInfo = createReduceInfo(inputs, -1);
    const program = packed ? new RMSProgram16(deviceInfo, reduceInfo) : new RMSProgram32(deviceInfo, reduceInfo);

    assertShapesMatch(pGamma.shape, [pX.shape[pX.shape.length - 1]], 'Error in RMSNorm: ');
    if (x.shape.length !== 3) {
        throw new Error(`rmsNormGPU: input rank ${x.shape.length} not supported, only rank 3 is supported`);
    }
    if (reduceInfo.inSize !== pX.shape[pX.shape.length - 1]) {
        throw new Error(
            `rmsNormGPU: reduction size ${reduceInfo.inSize} does not match expected size ${
                pX.shape[pX.shape.length - 1]
            }`
        );
    }
    if (reduceInfo.batchSize !== x.shape[0] * x.shape[1]) {
        throw new Error(
            `rmsNormGPU: batch size ${reduceInfo.batchSize} does not match expected size ${x.shape[0] * x.shape[1]}`
        );
    }

    const result = reduce(program, inputs, backend);

    if (packed && !packedX) {
        pX.dispose();
    }
    if (packed && !packedGamma) {
        pGamma.dispose();
    }

    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'RMSNorm',
    backendName: 'webgpu',
    kernelFunc: rmsNormGPU,
};

registerKernel(kernelConfig);
