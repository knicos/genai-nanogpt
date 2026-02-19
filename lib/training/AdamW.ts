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
import { dispose, engine, Optimizer, scalar, Scalar, tidy, zeros } from '@tensorflow/tfjs-core';
import { OptimizerVariable } from '@tensorflow/tfjs-core/dist/optimizers/optimizer';
import { ConfigDict, Serializable, SerializableConstructor } from '@tensorflow/tfjs-core/dist/serialization';
import { NamedTensor, NamedVariableMap } from '@tensorflow/tfjs-core/dist/tensor_types';
import LRScheduler, { LRSchedulerConfig } from './LRScheduler';
import { clipScale } from '@base/ops/globalNorm';

export interface AdamWOptimizerConfig extends LRSchedulerConfig {
    learningRate: number;
    beta1: number;
    beta2: number;
    epsilon?: number;
    weightDecay: number;
    lossScaling: number;
    clipNorm?: number;
}

export class AdamWOptimizer extends Optimizer {
    public readonly className = 'AdamW';

    private accBeta1 = 0;
    private accBeta2 = 0;
    private accumulatedMoments: OptimizerVariable[] = [];
    protected learningRate: number;
    protected beta1: number;
    protected beta2: number;
    protected lossScaling: number;
    protected weightDecay: number;
    protected epsilon: number | null = null;
    protected lrScheduler: LRScheduler;
    protected clipNorm?: number;

    constructor(private config: AdamWOptimizerConfig) {
        super();
        this.accBeta1 = config.beta1;
        this.accBeta2 = config.beta2;
        this.learningRate = config.learningRate;
        this.beta1 = config.beta1;
        this.beta2 = config.beta2;
        this.weightDecay = config.weightDecay;
        this.lossScaling = config.lossScaling;
        this.clipNorm = config.clipNorm;
        if (config.epsilon === null || config.epsilon === undefined) {
            this.epsilon = engine().backend.epsilon();
        } else {
            this.epsilon = config.epsilon;
        }

        this.lrScheduler = new LRScheduler(config.learningRate, config);
    }

    get lr(): number {
        return this.learningRate;
    }

    updateConfig(newConfig: Partial<AdamWOptimizerConfig>) {
        const config = { ...this.config, ...newConfig };
        this.learningRate = config.learningRate;
        this.beta1 = config.beta1;
        this.beta2 = config.beta2;
        this.weightDecay = config.weightDecay;
        this.lossScaling = config.lossScaling;
        this.epsilon = config.epsilon ?? this.epsilon;
        this.clipNorm = config.clipNorm;

        this.lrScheduler.updateConfig(config, config.learningRate);
    }

    applyGradients(variableGradients: NamedVariableMap | NamedTensor[]) {
        const lr = this.lrScheduler.getNextLR();
        this.learningRate = lr;

        const varNames = Array.isArray(variableGradients)
            ? variableGradients.map((v) => v.name)
            : Object.keys(variableGradients);
        tidy(() => {
            const oneMinusAccBeta1 = 1 - this.accBeta1;
            const oneMinusAccBeta2 = 1 - this.accBeta2;

            let scaling: Scalar;
            if (this.clipNorm !== undefined) {
                const grads = varNames.map((name, i) => {
                    const gradient = Array.isArray(variableGradients)
                        ? variableGradients[i].tensor
                        : variableGradients[name];
                    return gradient;
                });
                scaling = clipScale(grads, 1 / this.lossScaling, this.clipNorm);
            } else {
                scaling = scalar(1 / this.lossScaling);
            }

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

                const newMoments = adamMoments(moments, gradient, this.beta1, this.beta2, scaling);
                moments.assign(newMoments);

                const newValue = adamAdjust(
                    newMoments,
                    value,
                    oneMinusAccBeta1,
                    oneMinusAccBeta2,
                    this.epsilon ?? 1e-8,
                    this.learningRate,
                    // Only apply weight decay if the variable is multi-dimensional (e.g. weights, not biases)
                    value.shape.length > 1 ? this.weightDecay : 0
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
