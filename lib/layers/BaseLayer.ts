import { GPTConfig } from '@base/models/config';
import MemoryProfiler from '@base/utilities/profile';
import RoPECache from './RoPECache';
import { customGrad, engine, grads, GradSaveFunc, Tensor, Variable } from '@tensorflow/tfjs-core';
import WeightStore from './WeightStore';

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
    public weightStore: WeightStore;
    public readonly children: BaseLayer[] = [];
    private profiler?: MemoryProfiler;

    constructor(config: GPTConfig, parent?: BaseLayer) {
        this.config = config;
        this.parent = parent;
        if (this.parent) {
            this.parent.children.push(this);
            this.weightStore = this.parent.weightStore;
        } else {
            this.weightStore = new WeightStore();
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
        this.weightStore.addVariable(name, variable);
    }

    get variables(): Variable[] {
        return this.weightStore.variables;
    }

    get trainableVariables(): Variable[] {
        return this.weightStore.trainableVariables;
    }

    public getVariable(name: string): Tensor {
        return this.weightStore.getVariable(name);
    }

    public hasVariable(name: string): boolean {
        return this.weightStore.hasVariable(name);
    }

    public setVariable(name: string, variable: Variable) {
        this.weightStore.setVariable(name, variable);
    }

    public dispose(): void {
        this.weightStore.dispose();
    }

    protected build(): void {
        // Nothing to do by default.
    }
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
