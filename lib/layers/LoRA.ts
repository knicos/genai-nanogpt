import { randomNormal, Scalar, scalar, tidy, variable, zeros } from '@tensorflow/tfjs-core';
import WeightStore from './WeightStore';
import picomatch from 'picomatch';

export default class LoRA {
    private weightStore: WeightStore;
    public readonly alpha: number;
    public readonly rank: number;
    public readonly variables: Set<string>;
    private scale: Scalar;
    public readonly name: string;

    constructor(name: string, weightStore: WeightStore, alpha: number, rank: number, variables: string[]) {
        this.name = name;
        this.weightStore = weightStore;
        this.alpha = alpha;
        this.rank = rank;

        const isMatch = picomatch(variables);
        const selectedVariables = weightStore.variableNames.filter(
            (name) => isMatch(name) && !name.endsWith('_loraA') && !name.endsWith('_loraB')
        );
        this.variables = new Set(selectedVariables);

        this.scale = scalar(alpha / rank);

        // Initialize LoRA variables
        this.variables.forEach((varName) => {
            const originalVar = this.weightStore.getRawVariable(varName);
            const [outDim, inDim] = originalVar.shape;
            const loraAName = `${varName}_${this.name}_loraA`;
            const loraBName = `${varName}_${this.name}_loraB`;

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
    }

    attach() {
        if (this.weightStore.onWeightRead) {
            throw new Error('LoRA cannot be applied to a WeightStore that already has a onWeightRead hook.');
        }

        // Hook into weight reads to apply LoRA adjustments
        this.weightStore.onWeightRead = (name, variable) => {
            if (this.variables.has(name)) {
                return tidy(() => {
                    const loraA = this.weightStore.getRawVariable(`${name}_${this.name}_loraA`);
                    const loraB = this.weightStore.getRawVariable(`${name}_${this.name}_loraB`);
                    // Apply LoRA adjustment: W + alpha * A @ B
                    return variable.add(loraA.matMul(loraB).mul(this.scale));
                });
            }
            return variable;
        };

        // Disable training of all other weights
        this.weightStore.setTrainable([`*_${this.name}_loraA`, `*_${this.name}_loraB`]);
    }

    merge() {
        this.variables.forEach((varName) => {
            const originalVar = this.weightStore.getRawVariable(varName);
            const loraA = this.weightStore.getRawVariable(`${varName}_${this.name}_loraA`);
            const loraB = this.weightStore.getRawVariable(`${varName}_${this.name}_loraB`);
            const mergedVar = tidy(() => originalVar.add(loraA.matMul(loraB).mul(this.scale)));
            originalVar.assign(mergedVar);
            mergedVar.dispose();
        });
    }

    detach() {
        this.weightStore.onWeightRead = undefined;
        this.weightStore.setTrainable(['*']);
    }

    dispose() {
        this.detach();
        this.scale.dispose();
        this.variables.forEach((varName) => {
            const loraAName = `${varName}_${this.name}_loraA`;
            const loraBName = `${varName}_${this.name}_loraB`;
            this.weightStore.getRawVariable(loraAName).dispose();
            this.weightStore.getRawVariable(loraBName).dispose();
            this.weightStore.deleteVariable(loraAName);
            this.weightStore.deleteVariable(loraBName);
        });
        this.variables.clear();
    }
}
