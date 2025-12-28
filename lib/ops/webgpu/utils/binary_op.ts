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

// Nick: Modified from tfjs-backend-webgpu/dist/binary_op.ts to support 16-bit float ops

import { WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { BinaryOpType, getBinaryOpString } from '@tensorflow/tfjs-backend-webgpu/dist/binary_op_util';
import { backend_util, util } from '@tensorflow/tfjs-core';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';

export { BinaryOpType };

export class BinaryOpProgram implements WebGPUProgram {
    dispatch: [number, number, number];
    dispatchLayout: { x: number[] };
    outputComponent: number;
    op: BinaryOpType;
    outputShape: number[];
    shaderKey: string;
    size = true;
    variableNames = ['A', 'B'];
    workgroupSize: [number, number, number];
    variableComponents: number[];

    constructor(op: BinaryOpType, aShape: number[], bShape: number[]) {
        this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.op = op;

        const aDivisibleBy4 = aShape.length > 0 && aShape[aShape.length - 1] % 4 === 0;
        const bDivisibleBy4 = bShape.length > 0 && bShape[bShape.length - 1] % 4 === 0;
        if (aDivisibleBy4 && bDivisibleBy4) {
            this.outputComponent = 4;
            this.variableComponents = [4, 4];
        } else if (
            (aDivisibleBy4 && (util.isScalarShape(bShape) || bShape[bShape.length - 1] === 1)) ||
            (bDivisibleBy4 && (util.isScalarShape(aShape) || aShape[aShape.length - 1] === 1))
        ) {
            //this.outputComponent = 4;
            //.variableComponents = aDivisibleBy4 ? [4, 1] : [1, 4];
            throw new Error('Cannot broadcast 16-bit float binary ops with mixed vector sizes');
        } else {
            //this.outputComponent = 1;
            //this.variableComponents = [1, 1];
            throw new Error('16-bit float binary ops require inner dimension to be multiple of 4');
        }

        this.shaderKey = `binary_${op}_${this.variableComponents}`;
        // TODO(jiajia.qin@intel.com): Heuristically select a good work group
        // size.
        this.workgroupSize = [128, 1, 1];

        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [
            this.outputComponent,
            1,
            1,
        ]);
    }

    getUserCode(): string {
        const dType = this.outputComponent === 4 ? 'vec4<f32>' : 'f32';
        const opFnStr = `
    fn binaryOperation(a : ${dType}, b : ${dType}) -> ${dType} {
      ${getBinaryOpString(this.op, this.outputComponent === 4)}
    };
    `;

        // NOTE: Always assumes vectors of 4 for inputs and output for 16-bit float ops.

        const userCode = `
       ${opFnStr}
       ${main('index')} {
         if (index < uniforms.size) {
            let a = A[index];
            let b = B[index];

            let v4a1 = vec4<f32>(
                unpack2x16float(u32(a.x)),
                unpack2x16float(u32(a.y))
            );
            let v4a2 = vec4<f32>(
                unpack2x16float(u32(a.z)),
                unpack2x16float(u32(a.w))
            );
            let v4b1 = vec4<f32>(
                unpack2x16float(u32(b.x)),
                unpack2x16float(u32(b.y))
            );
            let v4b2 = vec4<f32>(
                unpack2x16float(u32(b.z)),
                unpack2x16float(u32(b.w))
            );

            let v4res1 = binaryOperation(v4a1, v4b1);
            let v4res2 = binaryOperation(v4a2, v4b2);

            let res = vec4<i32>(
                i32(pack2x16float(v4res1.xy)),
                i32(pack2x16float(v4res1.zw)),
                i32(pack2x16float(v4res2.xy)),
                i32(pack2x16float(v4res2.zw))
            );

            result[index] = res;
         }
       }
       `;

        return userCode;
    }
}
