/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

// Nicolas Pope: Modified to fuse the operations

import { adamAdjust } from '@base/ops/adamAdjust';
import { adamMoments } from '@base/ops/adamMoments';
import { dispose, engine, Optimizer, tidy, zeros } from '@tensorflow/tfjs-core';
import { OptimizerVariable } from '@tensorflow/tfjs-core/dist/optimizers/optimizer';
import { ConfigDict, Serializable, SerializableConstructor } from '@tensorflow/tfjs-core/dist/serialization';
import { NamedTensor, NamedVariableMap } from '@tensorflow/tfjs-core/dist/tensor_types';

export class AdamOptimizer extends Optimizer {
    /** @nocollapse */
    static get className() {
        // Name matters for Python compatibility.
        // This is a getter instead of a property because when it's a property, it
        // prevents the entire class from being tree-shaken.
        return 'Adam';
    }

    private accBeta1: number = 0;
    private accBeta2: number = 0;
    private accumulatedMoments: OptimizerVariable[] = [];

    constructor(
        protected learningRate: number,
        protected beta1: number,
        protected beta2: number,
        protected lossScaling: number,
        protected epsilon: number | null = null
    ) {
        super();
        this.accBeta1 = beta1;
        this.accBeta2 = beta2;

        if (epsilon === null) {
            this.epsilon = engine().backend.epsilon();
        }
    }

    applyGradients(variableGradients: NamedVariableMap | NamedTensor[]) {
        const varNames = Array.isArray(variableGradients)
            ? variableGradients.map((v) => v.name)
            : Object.keys(variableGradients);
        tidy(() => {
            const oneMinusAccBeta1 = 1 - this.accBeta1;
            const oneMinusAccBeta2 = 1 - this.accBeta2;

            varNames.forEach((name, i) => {
                const value = engine().registeredVariables[name];
                const trainable = false;
                if (this.accumulatedMoments[i] == null) {
                    this.accumulatedMoments[i] = {
                        originalName: `${name}/m`,
                        variable: tidy(() => zeros([...value.shape, 2]).variable(trainable)),
                    };
                }

                const gradient = Array.isArray(variableGradients)
                    ? variableGradients[i].tensor
                    : variableGradients[name];
                if (gradient == null) {
                    return;
                }

                const moments = this.accumulatedMoments[i].variable;

                const newMoments = adamMoments(moments, gradient, this.beta1, this.beta2, this.lossScaling);
                moments.assign(newMoments);

                const newValue = adamAdjust(
                    newMoments,
                    value,
                    oneMinusAccBeta1,
                    oneMinusAccBeta2,
                    this.epsilon ?? 1e-8,
                    this.learningRate
                );
                value.assign(newValue);
            });

            this.accBeta1 = this.accBeta1 * this.beta1;
            this.accBeta2 = this.accBeta2 * this.beta2;
        });
        this.incrementIterations();
    }

    override dispose(): void {
        if (this.accumulatedMoments != null) {
            dispose(this.accumulatedMoments.map((v) => v.variable));
        }
    }

    override async getWeights(): Promise<NamedTensor[]> {
        // Order matters for Python compatibility.
        const variables: OptimizerVariable[] = [...this.accumulatedMoments];
        return [await this.saveIterations()].concat(
            variables.map((v) => ({ name: v.originalName, tensor: v.variable }))
        );
    }

    override async setWeights(weightValues: NamedTensor[]): Promise<void> {
        weightValues = await this.extractIterations(weightValues);
        tidy(() => {
            this.accBeta1 = Math.pow(this.beta1, this.iterations_ + 1);
            this.accBeta2 = Math.pow(this.beta2, this.iterations_ + 1);
        });

        const variableCount = weightValues.length / 2;
        const trainable = false;
        this.accumulatedMoments = weightValues.slice(0, variableCount).map((v) => ({
            originalName: v.name,
            variable: v.tensor.variable(trainable),
        }));
    }

    getConfig(): ConfigDict {
        return {
            learningRate: this.learningRate,
            beta1: this.beta1,
            beta2: this.beta2,
            epsilon: this.epsilon,
        };
    }

    /** @nocollapse */
    static override fromConfig<T extends Serializable>(cls: SerializableConstructor<T>, config: ConfigDict): T {
        return new cls(config['learningRate'], config['beta1'], config['beta2'], config['epsilon']);
    }
}
