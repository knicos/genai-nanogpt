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
import { WebGPUBackend, WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { reshape16 } from '../reshape16';
import { PackedTensorInfo } from '@base/patches/PackedTensor';

class SoftmaxProgram implements WebGPUProgram {
    variableNames = ['logits'];
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number];

    constructor(outputShape: number[]) {
        this.outputShape = outputShape; // [rows, cols]
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = [this.outputShape[0], 1, 1];
        if (this.outputShape[1] >= 4096) {
            this.workgroupSize = [256, 1, 1];
        } else if (this.outputShape[1] < 64) {
            this.workgroupSize = [32, 1, 1];
        } else {
            this.workgroupSize = [64, 1, 1];
        }
        this.shaderKey = 'softmax16';
    }

    getUserCode(): string {
        const userCode = `
        var<workgroup> buf : array<f32, ${this.workgroupSize[0]}>;
        const blockSize = ${this.workgroupSize[0]};
        ${main('index')} {
            let row = index / blockSize;
            let tid = i32(localId.x);
            let cols = uniforms.outShape[1];
            let rowIdx = row * cols;

            var threadMax = -3.402823e+38f;
            for (var col = tid; col < cols; col += blockSize) {
                let value = unpack2x16float(u32(logits[rowIdx + col]));
                threadMax = max(threadMax, max(value.x, value.y));
            }
            buf[tid] = threadMax;
            workgroupBarrier();

            for (var currSize = blockSize >> 1;  currSize > 0; currSize = currSize >> 1) {
                if (tid < currSize) {
                    buf[tid] = max(buf[tid], buf[tid + currSize]);
                }
                workgroupBarrier();
            }

            let rowMaxShared: f32 = buf[0];
            workgroupBarrier();

            var threadSum = 0.0f;
            for (var col = tid; col < cols; col += blockSize) {
                let value = unpack2x16float(u32(logits[rowIdx + col]));
                let subExp = exp(value.x - rowMaxShared);
                threadSum += subExp;
                let subExpY = exp(value.y - rowMaxShared);
                threadSum += subExpY;
            }
            buf[tid] = threadSum;
            workgroupBarrier();

            for (var currSize = blockSize >> 1;  currSize > 0; currSize = currSize >> 1) {
                if (tid < currSize) {
                    buf[tid] = buf[tid] + buf[tid + currSize];
                }
                workgroupBarrier();
            }

            let rowSumShared: f32 = buf[0];

            for (var col = tid; col < cols; col += blockSize) {
                let value = unpack2x16float(u32(logits[rowIdx + col]));
                let value1: f32 = exp(value.x - rowMaxShared) / rowSumShared;
                let value2: f32 = exp(value.y - rowMaxShared) / rowSumShared;
                result[rowIdx + col] = i32(pack2x16float(vec2<f32>(value1, value2)));
            }
        }
      `;
        return userCode;
    }
}

function softmax(args: { inputs: SoftmaxInputs; backend: WebGPUBackend; attrs: SoftmaxAttrs }): TensorInfo {
    const { inputs, backend, attrs } = args;
    const { logits } = inputs;
    const { dim } = attrs;

    const logitsReshaped = reshape(logits as Tensor, [
        util.sizeFromShape(logits!.shape) / logits!.shape[dim],
        logits!.shape[dim],
    ]);
    const program = new SoftmaxProgram(logitsReshaped.shape);
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
