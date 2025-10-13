/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

// Nicolas Pope: Patched to fix Firefox bug

import { KernelConfig, KernelFunc, Log, registerKernel } from '@tensorflow/tfjs-core';

import {
    CHECK_NAN_SNIPPET_UNARY,
    unaryKernelFunc,
} from '@tensorflow/tfjs-backend-webgl/dist/kernel_utils/kernel_funcs_utils';
import { logImplCPU, SimpleUnaryKernelImplCPU } from '@tensorflow/tfjs-backend-webgl/dist/kernel_utils/shared';

// Windows chrome return 0 if the input is negative value. We will specifically
// return NaN if the input is 0 to solve compatiblity issue.
const LOG =
    CHECK_NAN_SNIPPET_UNARY +
    `
  return x < 0.0 ? NAN : log(x);
`;

const LOG_PACKED = `
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : (x.r < 0.0 ? NAN : result.r);
  result.g = isNaN.g ? x.g : (x.g < 0.0 ? NAN : result.g);
  result.b = isNaN.b ? x.b : (x.b < 0.0 ? NAN : result.b);
  result.a = isNaN.a ? x.a : (x.a < 0.0 ? NAN : result.a);
  return result;
`;

const log = unaryKernelFunc({
    opSnippet: LOG,
    packedOpSnippet: LOG_PACKED,
    cpuKernelImpl: logImplCPU as SimpleUnaryKernelImplCPU,
});

const logConfig: KernelConfig = {
    kernelName: Log,
    backendName: 'webgl',
    kernelFunc: log as unknown as KernelFunc,
};

registerKernel(logConfig);
