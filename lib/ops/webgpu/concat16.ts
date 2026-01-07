/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

// Nick: Adapted from tfjs Concat to support 16-bit packed tensors

import {
    backend_util,
    ConcatAttrs,
    ConcatInputs,
    KernelConfig,
    KernelFunc,
    registerKernel,
    TensorInfo,
    util,
} from '@tensorflow/tfjs-core';
import { getMainHeaderString as main, WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu';
import { reshape } from '@tensorflow/tfjs-backend-webgpu/dist/kernels/Reshape';

export class ConcatProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey: string;
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    variableNames: string[];
    uniforms = '';
    workPerThread = 1;
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    offsetLength: number;

    constructor(shapes: Array<[number, number]>) {
        this.outputShape = backend_util.computeOutShape(shapes, 1 /* axis */) as [number, number];
        this.variableNames = shapes.map((_, i) => `T${i}`);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [
            this.workPerThread,
            1,
            1,
        ]);

        this.offsetLength = shapes.length - 1;
        for (let i = 0; i < this.offsetLength; i++) {
            this.uniforms += `offset${i} : i32,`;
        }
        this.shaderKey = 'concat16';
    }

    getUserCode(): string {
        const snippets: string[] = [];
        if (this.offsetLength > 0) {
            snippets.push(
                `if (yC < uniforms.offset0){ result[getIndexFromCoords2D(coords, uniforms.outShape)] = T0[getIndexFromCoords2D(vec2<i32>(yR, yC), uniforms.t0Shape)]; }`
            );
            for (let i = 1; i < this.offsetLength; i++) {
                snippets.push(
                    `else if (yC < uniforms.offset${[i]}){ ` +
                        `result[getIndexFromCoords2D(coords, uniforms.outShape)] = T${i}[getIndexFromCoords2D(vec2<i32>(yR, yC - uniforms.offset${
                            i - 1
                        }), uniforms.t${i}Shape)]; }`
                );
            }
            const lastIndex = this.offsetLength;
            const lastShiftIndex = this.offsetLength - 1;
            snippets.push(
                `else { result[getIndexFromCoords2D(coords, uniforms.outShape)] = T${lastIndex}[getIndexFromCoords2D(vec2<i32>(yR, yC - uniforms.offset${lastShiftIndex}), uniforms.t${lastIndex}Shape)]; }`
            );
        } else {
            snippets.push(
                `result[getIndexFromCoords2D(coords, uniforms.outShape)] = T0[getIndexFromCoords2D(vec2<i32>(yR, yC), uniforms.t0Shape)];`
            );
        }

        const userCode = `
      ${main('index')} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            let yR = coords.x;
            let yC = coords.y;

            ${snippets.join('\n        ')}
          }
        }
      }
    `;
        return userCode;
    }
}

function concatImpl(inputs: ConcatInputs, axis: number, backend: WebGPUBackend): TensorInfo {
    // There is a storage buffer limitation in compute stage, one for output so
    // the maximum for input is limits.maxStorageBuffersPerShaderStage - 1
    const maxInputNum = backend.device.limits.maxStorageBuffersPerShaderStage - 1;
    if (inputs.length > maxInputNum) {
        const reducedInputs = [];
        for (let i = 0; i < inputs.length; i += maxInputNum) {
            const subArray = inputs.slice(i, i + maxInputNum);
            reducedInputs.push(concatImpl(subArray, axis, backend));
        }
        const result = concatImpl(reducedInputs, axis, backend);

        for (const i of reducedInputs) {
            backend.disposeData(i.dataId);
        }

        return result;
    }

    const { tensors2D, outShape } = computeTensors2D(inputs, axis, backend);
    const shapes = tensors2D.map((t) => t.shape as [number, number]);
    const program = new ConcatProgram(shapes);

    const uniformData: Array<{ type: string; data: number[] }> = [];
    const offsets: number[] = new Array(shapes.length - 1);
    if (offsets.length > 0) {
        offsets[0] = shapes[0][1];
        uniformData.push({ type: 'int32', data: [offsets[0]] });
        for (let i = 1; i < offsets.length; i++) {
            offsets[i] = offsets[i - 1] + shapes[i][1];
            uniformData.push({ type: 'int32', data: [offsets[i]] });
        }
    }

    const res = backend.runWebGPUProgram(program, tensors2D, tensors2D[0].dtype, uniformData);
    tensors2D.forEach((r) => backend.disposeData(r.dataId));

    const reshapedResult = reshape({ inputs: { x: res }, backend, attrs: { shape: outShape } });
    backend.disposeData(res.dataId);
    return reshapedResult;
}

function computeTensors2D(inputs: ConcatInputs, axis: number, backend: WebGPUBackend) {
    const outShape = backend_util.computeOutShape(
        inputs.map((t) => t.shape),
        axis
    );
    const tensors2D = inputs.map((t) =>
        reshape({
            inputs: { x: t },
            backend,
            attrs: {
                shape: [util.sizeFromShape(t.shape.slice(0, axis)), util.sizeFromShape(t.shape.slice(axis))],
            },
        })
    );

    return { tensors2D, outShape };
}

function concat(args: { inputs: ConcatInputs; attrs: ConcatAttrs; backend: WebGPUBackend }): TensorInfo {
    const { inputs, backend, attrs } = args;
    const { axis } = attrs;

    const $axis = util.parseAxisParam(axis, inputs[0].shape)[0];

    const shapes = inputs.map((t) => t.shape);
    backend_util.assertParamsConsistent(shapes, $axis);

    const outShape = backend_util.computeOutShape(
        inputs.map((t) => t.shape),
        $axis
    );
    if (util.sizeFromShape(outShape) === 0) {
        return backend.makeTensorInfo(outShape, inputs[0].dtype, []);
    }

    // Keep only non-empty tensors (ignore tensors with 0 in their shape).
    const $inputs = inputs.filter((t) => util.sizeFromShape(t.shape) > 0);
    /*if ($inputs.length === 1) {
        return identity({ inputs: { x: $inputs[0] }, backend });
    }*/

    return concatImpl($inputs, $axis, backend);
}

export const concatConfig: KernelConfig = {
    kernelName: 'Concat16',
    backendName: 'webgpu',
    kernelFunc: concat as unknown as KernelFunc,
};

registerKernel(concatConfig);
