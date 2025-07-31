import type TF from '@tensorflow/tfjs';
import { GPTConfig } from './config';
import CausalSelfAttention from './CausalSelfAttention';
import MLP from './MLP';

// Transformer block
export default class Block {
    private ln1: TF.layers.Layer;
    private attn: CausalSelfAttention;
    private ln2: TF.layers.Layer;
    private mlp: MLP;
    private tf: typeof TF;
    private index: number;
    private _trainable: boolean = false;
    public skipped: boolean = false;

    constructor(tf: typeof TF, index: number, config: GPTConfig) {
        this.tf = tf;
        this.index = index;
        this.ln1 = this.tf.layers.layerNormalization({
            axis: -1,
            epsilon: 1e-5,
            center: config.biasInLayerNorm,
            scale: true,
            name: `attention_layer_norm_${this.index}`,
        });

        this.attn = new CausalSelfAttention(this.tf, this.index, config);

        this.ln2 = tf.layers.layerNormalization({
            axis: -1,
            epsilon: 1e-5,
            center: config.biasInLayerNorm,
            scale: true,
            name: `mlp_layer_norm_${this.index}`,
        });

        this.mlp = new MLP(this.tf, this.index, config);

        this.trainable = true; // Default to trainable
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
        map.set(`block_${this.index}_ln1`, this.ln1.getWeights());
        map.set(`block_${this.index}_ln2`, this.ln2.getWeights());
    }

    loadWeights(weights: Map<string, TF.Tensor[]>): void {
        this.attn.loadWeights(weights);
        this.mlp.loadWeights(weights);
        this.ln1.setWeights(weights.get(`block_${this.index}_ln1`) || []);
        this.ln2.setWeights(weights.get(`block_${this.index}_ln2`) || []);
    }

    call(x: TF.Tensor, training = false): TF.Tensor {
        if (this.skipped) {
            return x; // Skip this block if marked as skipped
        }
        return this.tf.tidy(() => {
            // Pre-normalization residual connections
            const norm1 = this.ln1.apply(x) as TF.Tensor;
            const attnOut = this.attn.call(norm1, training);
            const residual1 = x.add(attnOut);

            const norm2 = this.ln2.apply(residual1) as TF.Tensor;
            const mlpOut = this.mlp.call(norm2, training);
            const residual2 = residual1.add(mlpOut);

            return residual2;
        });
    }
}
