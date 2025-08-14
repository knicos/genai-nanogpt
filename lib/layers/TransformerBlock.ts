import type TF from '@tensorflow/tfjs';
import { GPTConfig } from '../config';
import CausalSelfAttention, { KVCache } from './CausalSelfAttention';
import MLP from './MLP';
import RoPECache from './RoPECache';
import RMSNorm from './RMSNorm';

// Transformer block
export default class Block {
    private ln1: RMSNorm;
    private attn: CausalSelfAttention;
    private ln2: RMSNorm;
    private mlp: MLP;
    private tf: typeof TF;
    private index: number;
    private _trainable: boolean = true;
    public skipped: boolean = false;

    constructor(tf: typeof TF, index: number, config: GPTConfig, ropeCache?: RoPECache) {
        this.tf = tf;
        this.index = index;

        this.ln1 = new RMSNorm(tf, [config.nEmbed], 1e-8, `block_${this.index}_rms1`);

        this.attn = new CausalSelfAttention(this.tf, this.index, config, ropeCache);

        this.ln2 = new RMSNorm(tf, [config.nEmbed], 1e-8, `block_${this.index}_rms2`);

        this.mlp = new MLP(this.tf, this.index, config);
    }

    get variables(): TF.Variable[] {
        return [
            ...this.ln1.trainableWeights.map((v) => v as TF.Variable),
            ...this.attn.variables,
            ...this.ln2.trainableWeights.map((v) => v as TF.Variable),
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

    saveWeights(map: Map<string, TF.Tensor[]>): void {
        this.attn.saveWeights(map);
        this.mlp.saveWeights(map);
        map.set(`block_${this.index}_rms1`, this.ln1.getWeights());
        map.set(`block_${this.index}_rms2`, this.ln2.getWeights());
    }

    loadWeights(weights: Map<string, TF.Tensor[]>): void {
        this.attn.loadWeights(weights);
        this.mlp.loadWeights(weights);
        this.ln1.setWeights(weights.get(`block_${this.index}_rms1`) || []);
        this.ln2.setWeights(weights.get(`block_${this.index}_rms2`) || []);
    }

    private getMLPOutput(x: TF.Tensor, training: boolean): TF.Tensor {
        const norm = this.ln2.apply(x) as TF.Tensor;
        const mlpOut = this.mlp.call(norm, training);
        const residual = x.add(mlpOut);
        return residual;
    }

    call(
        x: TF.Tensor,
        training = false,
        includeAttention = false,
        cache?: KVCache
    ): { output: TF.Tensor; attention?: TF.Tensor; cache?: KVCache } {
        return this.tf.tidy(() => {
            if (this.skipped) {
                return { output: x }; // Skip this block if marked as skipped
            }

            // Pre-normalization residual connections
            const norm1 = this.ln1.apply(x) as TF.Tensor;
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
