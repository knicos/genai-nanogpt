/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

// Nick: Patched to allow int32 packed gradients

import type { Tensor } from '@tensorflow/tfjs-core/dist/tensor';
import type { TapeNode } from '@tensorflow/tfjs-core/dist/tape';
import * as util from '@tensorflow/tfjs-core/dist/util';
import { isPackedTensor } from '@base/utilities/packed';

/**
 * Backpropagate gradients through the filtered TapeNodes.
 *
 * @param tensorAccumulatedGradientMap A map of Tensor to its gradient. This map
 * is mutated by this method.
 * @param filteredTape The filtered TapeNodes to backprop through.
 */
export function backpropagateGradients(
    tensorAccumulatedGradientMap: { [tensorId: number]: Tensor },
    filteredTape: TapeNode[],
    // eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
    tidy: (f: Function) => Tensor,
    add: (a: Tensor, b: Tensor) => Tensor
) {
    // Walk the tape backward and keep a map of Tensor to its gradient.
    for (let i = filteredTape.length - 1; i >= 0; i--) {
        const node = filteredTape[i];

        const dys: Tensor[] = [];
        node.outputs.forEach((o) => {
            const gradTensor = tensorAccumulatedGradientMap[o.id];
            if (gradTensor != null) {
                dys.push(gradTensor);
            } else {
                // This particular output is not in the back-propagation subgraph, so it
                // does not affect the final output, thus we put null for its dy.
                dys.push(null as unknown as Tensor);
            }
        });

        if (node.gradient == null) {
            throw new Error(`Cannot compute gradient: gradient function not found ` + `for ${node.kernelName}.`);
        }

        // Backprop dy through this node and accumulate gradients over the inputs.
        const inputGradients = node.gradient(dys);

        for (const inputName in node.inputs) {
            if (!(inputName in inputGradients)) {
                throw new Error(
                    `Cannot backprop through input ${inputName}. ` +
                        `Available gradients found: ${Object.keys(inputGradients)}.`
                );
            }

            // Call the gradient function.
            const dx = tidy(() => inputGradients[inputName]());

            // Nick: Allow packed int32 gradients
            const isPacked = isPackedTensor(dx);
            if (dx.dtype !== 'float32' && (!isPacked || dx.dtype !== 'int32')) {
                throw new Error(
                    `Error in gradient for op ${node.kernelName}. The gradient of input ` +
                        `${inputName} must have 'float32' dtype, but has '${dx.dtype}'`
                );
            }
            const x = node.inputs[inputName];
            if (!util.arraysEqual(dx.shape, x.shape)) {
                throw new Error(
                    `Error in gradient for op ${node.kernelName}. The gradient of input ` +
                        `'${inputName}' has shape '${dx.shape}', which does not match ` +
                        `the shape of the input '${x.shape}'`
                );
            }

            if (tensorAccumulatedGradientMap[x.id] == null) {
                tensorAccumulatedGradientMap[x.id] = dx;
            } else {
                const curGradient = tensorAccumulatedGradientMap[x.id];
                tensorAccumulatedGradientMap[x.id] = add(curGradient, dx);
                curGradient.dispose();
            }
        }
    }
}
