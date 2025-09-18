import { GPTConfig } from '../config';
import CausalSelfAttention, { KVCache } from './CausalSelfAttention';
import MLP from './MLP';
import RoPECache from './RoPECache';
import RMSNorm from './RMSNorm';
import MemoryProfiler from '@base/utilities/profile';
import BaseLayer from './BaseLayer';
import { Tensor, tidy, Variable } from '@tensorflow/tfjs-core';

// Transformer block
export default class Block extends BaseLayer {
    private ln1: RMSNorm;
    private attn: CausalSelfAttention;
    private ln2: RMSNorm;
    private mlp: MLP;
    private index: number;
    private _trainable: boolean = true;
    public skipped: boolean = false;

    constructor(index: number, config: GPTConfig, ropeCache?: RoPECache) {
        super();
        this.index = index;

        this.ln1 = new RMSNorm([config.nEmbed], 1e-8, `block_${this.index}_rms1`);

        this.attn = new CausalSelfAttention(this.index, config, ropeCache);

        this.ln2 = new RMSNorm([config.nEmbed], 1e-8, `block_${this.index}_rms2`);

        this.mlp = new MLP(this.index, config);
    }

    override setProfiler(value: MemoryProfiler | undefined): void {
        this._profiler = value;
        this.attn.setProfiler(value);
        this.mlp.setProfiler(value);
        this.ln1.setProfiler(value);
        this.ln2.setProfiler(value);
    }

    get variables(): Variable[] {
        return [
            ...this.ln1.trainableWeights.map((v) => v as Variable),
            ...this.attn.variables,
            ...this.ln2.trainableWeights.map((v) => v as Variable),
            ...this.mlp.variables,
        ];
    }

    get trainable(): boolean {
        return this._trainable;
    }

    set trainable(value: boolean) {
        this._trainable = value;
        this.ln1.trainable = value;
        this.ln2.trainable = value;
        this.attn.trainable = value;
        this.mlp.trainable = value;
    }

    saveWeights(map: Map<string, Tensor[]>): void {
        this.attn.saveWeights(map);
        this.mlp.saveWeights(map);
        map.set(`block_${this.index}_rms1`, this.ln1.getWeights());
        map.set(`block_${this.index}_rms2`, this.ln2.getWeights());
    }

    loadWeights(weights: Map<string, Tensor[]>): void {
        this.attn.loadWeights(weights);
        this.mlp.loadWeights(weights);
        this.ln1.setWeights(weights.get(`block_${this.index}_rms1`) || []);
        this.ln2.setWeights(weights.get(`block_${this.index}_rms2`) || []);
    }

    private getMLPOutput(x: Tensor, training: boolean): Tensor {
        const norm = this.ln2.apply(x) as Tensor;
        const mlpOut = this.mlp.call(norm, training);
        const residual = x.add(mlpOut);
        return residual;
    }

    call(
        x: Tensor,
        training = false,
        includeAttention = false,
        cache?: KVCache
    ): { output: Tensor; attention?: Tensor; cache?: KVCache } {
        return tidy(() => {
            if (this.skipped) {
                return { output: x }; // Skip this block if marked as skipped
            }

            // Pre-normalization residual connections
            const norm1 = this.ln1.apply(x) as Tensor;
            const attnOut = this.attn.call(norm1, training, includeAttention, cache);
            const residual1 = x.add(attnOut.output);

            return {
                output: this.getMLPOutput(residual1, training),
                attention: attnOut.attention,
                cache: attnOut.presentKV,
            };
        });
    }

    dispose() {
        this.ln1.dispose();
        this.attn.dispose();
        this.ln2.dispose();
        this.mlp.dispose();
    }
}
