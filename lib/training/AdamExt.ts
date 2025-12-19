import { mul, sub, scalar, engine, Variable } from '@tensorflow/tfjs-core';
import { NamedTensor, NamedVariableMap } from '@tensorflow/tfjs-core/dist/tensor_types';
import { AdamOptimizer } from './Adam';

interface AdamExtConfig {
    warmupSteps: number;
    decaySteps: number;
    minLearningRate: number;
    weightDecay?: number;
    lossScaling: number;
}

/**
 * Extended Adam optimizer with warmup, cosine decay, and optional weight decay.
 */
export default class AdamExt extends AdamOptimizer {
    private step: number = 0;
    private startLearningRate: number;

    constructor(learningRate: number, beta1: number, beta2: number, epsilon: number, private config: AdamExtConfig) {
        super(learningRate, beta1, beta2, epsilon, config.lossScaling);
        this.startLearningRate = learningRate;
    }

    get lr(): number {
        return this.learningRate;
    }

    private getAdjustedLearningRate(): number {
        this.step++;
        if (this.step < this.config.warmupSteps) {
            const warmupFactor = Math.min(1, (this.step + 1) / (this.config.warmupSteps + 1));
            const adjustedLearningRate = this.startLearningRate * warmupFactor;
            return adjustedLearningRate;
        }
        if (this.step > this.config.decaySteps) {
            return this.config.minLearningRate;
        }

        const decay_ratio = (this.step - this.config.warmupSteps) / (this.config.decaySteps - this.config.warmupSteps);
        const coeff = 0.5 * (1.0 + Math.cos(Math.PI * decay_ratio));
        return this.config.minLearningRate + coeff * (this.startLearningRate - this.config.minLearningRate);
    }

    override applyGradients(gradientsAndVariables: NamedVariableMap | NamedTensor[]): void {
        this.learningRate = this.getAdjustedLearningRate();

        // Call the parent method to apply gradients with the new learning rate
        super.applyGradients(gradientsAndVariables);

        // AdamW weight decay
        if (this.config.weightDecay && this.config.weightDecay > 0) {
            this.applyWeightDecay(gradientsAndVariables);
        }
    }

    private decayVariable(variable: Variable, decay: number, currentLR: number): void {
        if (variable && variable.shape.length >= 2) {
            const decayValue = mul(variable, scalar(currentLR * decay));
            variable.assign(sub(variable, decayValue));
            decayValue.dispose();
        }
    }

    private applyWeightDecay(gradientsAndVariables: NamedVariableMap | NamedTensor[]): void {
        const weightDecay = this.config.weightDecay!;
        const currentLR = this.learningRate;

        const vars = engine().registeredVariables;

        if (Array.isArray(gradientsAndVariables)) {
            gradientsAndVariables.forEach(({ name }) => {
                const variable = vars[name];
                this.decayVariable(variable, weightDecay, currentLR);
            });
        } else {
            Object.keys(gradientsAndVariables).forEach((name) => {
                const variable = vars[name];
                this.decayVariable(variable, weightDecay, currentLR);
            });
        }
    }
}
