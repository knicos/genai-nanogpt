import { clone, Tensor, variable, Variable } from '@tensorflow/tfjs-core';

export default class WeightStore {
    private _variables: Map<string, Variable | null> = new Map();
    private touchedVariables: Set<string> = new Set();

    saveWeights(map: Map<string, Tensor[]>) {
        this._variables.forEach((variable, name) => {
            if (variable && this.touchedVariables.has(name)) {
                map.set(name, [clone(variable)]);
            }
        });
    }

    loadWeights(weights: Map<string, Tensor[]>, reference: boolean, trainable = true): void {
        this._variables.forEach((vari, name) => {
            const weight = weights.get(name)?.[0];
            if (!weight) {
                // throw new Error(`Weights for ${name} not found`);
                return;
            }
            if (!vari) {
                this._variables.set(name, variable(weight, trainable, name));
            } else {
                vari.assign(weight);
            }

            // Weights loaded from a reference model are not to be saved again
            if (reference) {
                this.touchedVariables.delete(name);
            } else {
                this.touchedVariables.add(name);
            }
        });
    }

    public addVariable(name: string, variable?: Variable) {
        this._variables.set(name, variable || null);
    }

    get variables(): Variable[] {
        const myVariables = Array.from(this._variables.values()).filter((v): v is Variable => v !== null);
        return myVariables;
    }

    get variableNames(): string[] {
        return Array.from(this._variables.keys());
    }

    get trainableVariables(): Variable[] {
        const myVariables = Array.from(this._variables.values()).filter(
            (v): v is Variable => v !== null && v.trainable
        );
        return myVariables;
    }

    public getVariable(name: string): Variable {
        const vari = this._variables.get(name);
        if (!vari) {
            throw new Error(`Variable ${name} not found`);
        }
        return vari;
    }

    public hasVariable(name: string): boolean {
        return this._variables.get(name) !== null;
    }

    public setVariable(name: string, variable: Variable) {
        if (!this._variables.has(name)) {
            throw new Error(`Variable ${name} not found`);
        }
        this._variables.set(name, variable);
    }

    public touchVariables(names: string[]) {
        for (const name of names) {
            const vari = this._variables.get(name);
            if (vari) {
                this.touchedVariables.add(name);
            }
        }
    }

    public dispose(): void {
        this._variables.forEach((variable) => {
            variable?.dispose();
        });
        this._variables.clear();
    }
}
