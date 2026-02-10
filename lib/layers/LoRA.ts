import { randomNormal, Scalar, scalar, tidy, variable, zeros } from '@tensorflow/tfjs-core';
import WeightStore from './WeightStore';
import picomatch from 'picomatch';

export default class LoRA {
    private weightStore: WeightStore;
    public readonly alpha: number;
    public readonly rank: number;
    private variables: Set<string>;
    private scale: Scalar;

    constructor(weightStore: WeightStore, alpha: number, rank: number, variables: string[]) {
        this.weightStore = weightStore;
        this.alpha = alpha;
        this.rank = rank;

        const isMatch = picomatch(variables);
        const selectedVariables = weightStore.variableNames.filter((name) => isMatch(name));
        this.variables = new Set(selectedVariables);

        this.scale = scalar(alpha / rank);

        console.log('Attaching LoRA with rank', rank, 'and alpha', alpha, variables);

        if (this.weightStore.onWeightRead) {
            throw new Error('LoRA cannot be applied to a WeightStore that already has a onWeightRead hook.');
        }

        // Initialize LoRA variables
        this.variables.forEach((varName) => {
            const originalVar = this.weightStore.getRawVariable(varName);
            const [outDim, inDim] = originalVar.shape;
            const loraAName = `${varName}_loraA`;
            const loraBName = `${varName}_loraB`;

            if (originalVar.shape.length !== 2) {
                console.warn(
                    `LoRA currently only supports 2D weight matrices. Variable ${varName} has shape ${originalVar.shape}`
                );
                this.variables.delete(varName);
                return;
            }

            // Already loaded so skip.
            if (this.weightStore.hasVariable(loraAName) || this.weightStore.hasVariable(loraBName)) {
                return;
            }

            // Initialize LoRA A and B matrices
            this.weightStore.addVariable(
                loraAName,
                variable(randomNormal([outDim, this.rank], 0, 0.02), true, loraAName)
            );
            this.weightStore.addVariable(loraBName, variable(zeros([this.rank, inDim]), true, loraBName));
        });

        // Hook into weight reads to apply LoRA adjustments
        this.weightStore.onWeightRead = (name, variable) => {
            if (this.variables.has(name)) {
                return tidy(() => {
                    const loraA = this.weightStore.getRawVariable(`${name}_loraA`);
                    const loraB = this.weightStore.getRawVariable(`${name}_loraB`);
                    // Apply LoRA adjustment: W + alpha * A @ B
                    return variable.add(loraA.matMul(loraB).mul(this.scale));
                });
            }
            return variable;
        };

        // Disable training of all other weights
        this.weightStore.setTrainable(['*_loraA', '*_loraB']);
    }

    merge() {
        this.variables.forEach((varName) => {
            const originalVar = this.weightStore.getRawVariable(varName);
            const loraA = this.weightStore.getRawVariable(`${varName}_loraA`);
            const loraB = this.weightStore.getRawVariable(`${varName}_loraB`);
            const mergedVar = tidy(() => originalVar.add(loraA.matMul(loraB).mul(this.scale)));
            originalVar.assign(mergedVar);
            mergedVar.dispose();
        });
    }

    detach(merge = false) {
        if (merge) {
            this.merge();
        }
        this.variables.forEach((varName) => {
            const loraAName = `${varName}_loraA`;
            const loraBName = `${varName}_loraB`;
            this.weightStore.getRawVariable(loraAName).dispose();
            this.weightStore.getRawVariable(loraBName).dispose();
            this.weightStore.deleteVariable(loraAName);
            this.weightStore.deleteVariable(loraBName);
        });
        this.weightStore.onWeightRead = undefined;
        this.weightStore.setTrainable(['*']);
        this.variables.clear();
    }

    dispose() {
        this.detach();
        this.scale.dispose();
    }
}
