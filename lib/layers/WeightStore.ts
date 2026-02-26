import { clone, Tensor, variable, Variable } from '@tensorflow/tfjs-core';
import picomatch from 'picomatch';

export default class WeightStore {
    private _variables = new Map<string, Variable | null>();
    private touchedVariables = new Set<string>();

    // Hooks
    public onWeightRead?: (name: string, variable: Variable) => Tensor;

    saveWeights(map: Map<string, Tensor[]>) {
        this._variables.forEach((variable, name) => {
            if (variable && this.touchedVariables.has(name)) {
                map.set(name, [clone(variable)]);
            }
        });
    }

    loadWeights(weights: Map<string, Tensor[]>, reference: boolean, trainable = true): void {
        weights.forEach((weight, name) => {
            const w0 = weight[0];
            const vari = this._variables.get(name);

            if (!vari) {
                this._variables.set(name, variable(w0, trainable, name));
            } else {
                vari.assign(w0);
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

    public deleteVariable(name: string) {
        const vari = this._variables.get(name);
        if (vari) {
            vari.dispose();
        }
        this._variables.delete(name);
        this.touchedVariables.delete(name);
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

    public getRawVariable(name: string): Variable {
        const vari = this._variables.get(name);
        if (!vari) {
            throw new Error(`Variable ${name} not found`);
        }
        return vari;
    }

    public getVariable(name: string): Tensor {
        const vari = this._variables.get(name);
        if (!vari) {
            throw new Error(`Variable ${name} not found`);
        }
        if (this.onWeightRead) {
            return this.onWeightRead(name, vari);
        }
        return vari;
    }

    public setTrainable(names: string[]) {
        const isMatch = picomatch(names);
        this._variables.forEach((vari, name) => {
            if (vari) {
                vari.trainable = isMatch(name);
            }
        });
    }

    public hasVariable(name: string): boolean {
        return !!this._variables.get(name);
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
            console.log('Disposing variable', variable?.name);
            variable?.dispose();
        });
        this._variables.clear();
    }
}
