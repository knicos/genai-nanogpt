import { GPTConfig } from '@base/models/config';
import MemoryProfiler from '@base/utilities/profile';
import RoPECache from './RoPECache';
import { customGrad, engine, grads, GradSaveFunc, Tensor, variable, Variable } from '@tensorflow/tfjs-core';

export interface ForwardAttributes {
    training: boolean;
    checkpointing?: boolean;
    mixedPrecision?: boolean;
    ropeCache?: RoPECache;
    outputEmbeddings?: boolean;
    embeddings?: { name: string; tensor: Tensor }[];
}

export default abstract class BaseLayer<ATTR extends ForwardAttributes = ForwardAttributes> {
    public readonly parent?: BaseLayer;
    public readonly config: GPTConfig;
    private _variables: Map<string, Variable | null> = new Map();
    private _trainable: boolean = true;
    public readonly children: BaseLayer[] = [];
    private profiler?: MemoryProfiler;

    constructor(config: GPTConfig, parent?: BaseLayer) {
        this.config = config;
        this.parent = parent;
        if (this.parent) {
            this.parent.children.push(this);
        }
    }

    public getProfiler(): MemoryProfiler | undefined {
        return this.profiler;
    }

    public setProfiler(profiler: MemoryProfiler | null) {
        this.profiler = profiler ? profiler : undefined;
        this.children.forEach((child) => {
            child.setProfiler(profiler);
        });
    }

    public startMemory() {
        this.profiler?.startMemory();
    }

    public endMemory(label: string) {
        this.profiler?.endMemory(label);
    }

    public addVariable(name: string, variable?: Variable) {
        this._variables.set(name, variable || null);
    }

    get variables(): Variable[] {
        const myVariables = Array.from(this._variables.values()).filter((v): v is Variable => v !== null);
        const childVariables = this.children.flatMap((child) => child.variables);
        return [...myVariables, ...childVariables];
    }

    get trainableVariables(): Variable[] {
        const myVariables = Array.from(this._variables.values()).filter(
            (v): v is Variable => v !== null && v.trainable
        );
        const childVariables = this.children.flatMap((child) => child.trainableVariables);
        return [...myVariables, ...childVariables];
    }

    get trainable(): boolean {
        return this._trainable;
    }

    set trainable(value: boolean) {
        this._trainable = value;
        this._variables.forEach((variable) => {
            if (variable) {
                variable.trainable = value;
            }
        });
        this.children.forEach((child) => {
            child.trainable = value;
        });
    }

    public getVariable(name: string, recursive = false): Variable {
        const vari = this._variables.get(name);
        if (!vari && recursive) {
            for (const child of this.children) {
                const vari = child.getVariable(name, true);
                if (vari) {
                    return vari;
                }
            }
        }
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

    saveWeights(map: Map<string, Tensor[]>) {
        this._variables.forEach((variable, name) => {
            if (variable) {
                map.set(name, [variable.clone()]);
            }
        });
        this.children.forEach((child) => {
            child.saveWeights(map);
        });
    }

    loadWeights(weights: Map<string, Tensor[]>): void {
        this._variables.forEach((vari, name) => {
            const weight = weights.get(name)?.[0];
            if (!weight) {
                throw new Error(`Weights for ${name} not found`);
            }
            if (!vari) {
                this._variables.set(name, variable(weight, this._trainable));
            } else {
                vari.assign(weight);
            }
        });
        this.children.forEach((child) => {
            child.loadWeights(weights);
        });
    }

    public dispose(): void {
        this._variables.forEach((variable) => {
            variable?.dispose();
        });
        this._variables.clear();
    }

    protected build(): void {}
    protected dropout(x: Tensor): Tensor {
        return x;
    }

    abstract forward(attrs: ATTR, ...x: Tensor[]): Tensor | Tensor[];

    public call(attrs: ATTR, ...x: Tensor[]): Tensor | Tensor[] {
        this.build();
        const f = this.forward(attrs, ...x);
        if (attrs.training && f instanceof Tensor) {
            const out = this.dropout(f);
            if (out !== f) f.dispose();
            return out;
        } else {
            return f;
        }
    }

    public callCheckpoint(attrs: ATTR, ...x: Tensor[]): Tensor {
        this.build();
        const f = this.checkpointingFn(attrs, ...x);
        return f;
    }

    private checkpointingFn(attrs: ATTR, ...x: Tensor[]): Tensor {
        const vars = this.trainableVariables;
        const cp = customGrad((...args: (Tensor | GradSaveFunc)[]) => {
            const save = args[args.length - 1] as GradSaveFunc;
            const argTensors = args.slice(0, x.length) as Tensor[];
            // Forward pass
            // We need to pass fcVar and projVar to keep them in scope for the backward pass
            const output = this.forward(attrs, ...argTensors) as Tensor;

            save(argTensors);
            const gradFunc = (dy: Tensor, saved: Tensor[]) => {
                // Hack to allow nested grads calls
                const savedTape = engine().state.activeTape;
                engine().state.activeTape = [];

                // Recompute forward pass
                // We need to pass fcVar and projVar to keep them in scope for the backward pass
                const g = grads((...x: Tensor[]) => {
                    const output = this.forward(attrs, ...x.slice(0, argTensors.length)) as Tensor;
                    return output;
                })([...saved, ...vars], dy);

                // Restore tape
                engine().state.activeTape = savedTape;

                return g;
            };
            return { value: output, gradFunc };
        });

        const output = cp(...x, ...vars);
        if (attrs.training) {
            const out = this.dropout(output);
            if (out !== output) output.dispose();
            return out;
        } else {
            return output;
        }
    }
}
