/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// NICK: Modified to 16 bit packed floats

import {
    engine,
    KernelConfig,
    KernelFunc,
    registerKernel,
    reshape,
    SoftmaxAttrs,
    SoftmaxInputs,
    Tensor,
    TensorInfo,
    util,
} from '@tensorflow/tfjs-core';
import { reshape16 } from '../reshape16';
import { PackedTensorInfo } from '@base/patches/PackedTensor';
import SoftmaxProgram from './softmax16_program';
import WebGPUBackendPatch from '@base/patches/webgpu_backend';
import SoftmaxSubgroupProgram from './softmax16_subgroup_program';
import createDeviceInformation from './utils/deviceInfo';

function softmax(args: { inputs: SoftmaxInputs; backend: WebGPUBackendPatch; attrs: SoftmaxAttrs }): TensorInfo {
    const { inputs, backend, attrs } = args;
    const { logits } = inputs;
    const { dim } = attrs;

    const minSubgroupSize = backend.subgroupMinSize;
    const maxSubgroupSize = backend.subgroupMaxSize;
    const deviceInfo = createDeviceInformation(backend);
    const hasSubgroups = deviceInfo.subgroupsSupported;

    const logitsReshaped = reshape(logits as Tensor, [
        util.sizeFromShape(logits!.shape) / logits!.shape[dim],
        logits!.shape[dim],
    ]);
    const program = hasSubgroups
        ? new SoftmaxSubgroupProgram(logitsReshaped.shape, minSubgroupSize, maxSubgroupSize)
        : new SoftmaxProgram(logitsReshaped.shape);
    const res: PackedTensorInfo = backend.runWebGPUProgram(program, [logitsReshaped], 'int32');
    res.packed = true;
    logitsReshaped.dispose();
    const resTensor = engine().makeTensorFromTensorInfo(res);
    const resReshaped = reshape16(resTensor, logits!.shape);
    resTensor.dispose();
    return resReshaped;
}

const softmaxConfig: KernelConfig = {
    kernelName: 'Softmax16',
    backendName: 'webgpu',
    kernelFunc: softmax as unknown as KernelFunc,
};

registerKernel(softmaxConfig);
