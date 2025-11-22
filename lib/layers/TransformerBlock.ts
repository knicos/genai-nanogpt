import CausalSelfAttention, { AttentionScores, KVCache } from './CausalSelfAttention';
import MLP from './MLP';
import RMSNorm from './RMSNorm';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { Tensor, tidy } from '@tensorflow/tfjs-core';
import { GPTConfig } from '@base/models/config';

interface BlockAttributes extends ForwardAttributes {
    pastKV?: KVCache;
    seed?: number;
    attentionScores?: AttentionScores;
}

// Transformer block
export default class Block extends BaseLayer<BlockAttributes> {
    private ln1: RMSNorm;
    private attn: CausalSelfAttention;
    private ln2: RMSNorm;
    private mlp: MLP;
    private index: number;
    public skipped: boolean = false;

    constructor(index: number, config: GPTConfig, parent?: BaseLayer) {
        super(config, parent);
        this.index = index;

        this.ln1 = new RMSNorm(config, `block_${this.index}_rms1`, this);

        this.attn = new CausalSelfAttention(this.index, config, this);

        this.ln2 = new RMSNorm(config, `block_${this.index}_rms2`, this);

        this.mlp = new MLP(this.index, config, this);
    }

    private getMLPOutput(x: Tensor, training: boolean): Tensor {
        const norm = this.ln2.call({ training }, x) as Tensor;
        const mlpOut = this.mlp.call({ training }, norm) as Tensor;
        norm.dispose();
        const residual = x.add(mlpOut);
        x.dispose(); // Safe to dispose in this case
        mlpOut.dispose();
        return residual;
    }

    forward(attrs: BlockAttributes, x: Tensor): Tensor {
        return tidy(() => {
            if (this.skipped) {
                return x; // Skip this block if marked as skipped
            }

            // Pre-normalization residual connections
            const norm1 = this.ln1.call(attrs, x) as Tensor;
            const attnOut = this.attn.call(attrs, norm1) as Tensor;
            norm1.dispose();
            const residual1 = x.add(attnOut);
            attnOut.dispose();

            return this.getMLPOutput(residual1, attrs.training);
        });
    }

    dispose() {
        this.ln1.dispose();
        this.attn.dispose();
        this.ln2.dispose();
        this.mlp.dispose();
    }
}
