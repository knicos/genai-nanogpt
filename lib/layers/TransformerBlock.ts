import CausalSelfAttention, { AttentionScores, KVCache } from './CausalSelfAttention';
import MLP from './MLP';
import RMSNorm from './RMSNorm';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { keep, Tensor, tidy } from '@tensorflow/tfjs-core';
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

    private getMLPOutput(x: Tensor, attrs: BlockAttributes): Tensor {
        const norm = this.ln2.call({ training: attrs.training }, x) as Tensor;
        const mlpOut = this.mlp.call(attrs, norm) as Tensor;
        if (attrs.outputEmbeddings) {
            keep(norm);
            attrs.embeddings!.push({ name: `block_ln2_${this.index}`, tensor: norm });
        } else {
            norm.dispose();
        }
        const residual = x.add(mlpOut);
        x.dispose(); // Safe to dispose in this case
        if (attrs.outputEmbeddings) {
            keep(mlpOut);
            attrs.embeddings!.push({ name: `block_mlp_out_${this.index}`, tensor: mlpOut });
        } else {
            mlpOut.dispose();
        }
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
            if (attrs.outputEmbeddings) {
                keep(norm1);
                attrs.embeddings!.push({ name: `block_ln1_${this.index}`, tensor: norm1 });
            } else {
                norm1.dispose();
            }
            const residual1 = x.add(attnOut);
            if (attrs.outputEmbeddings) {
                keep(attnOut);
                attrs.embeddings!.push({ name: `block_attn_out_${this.index}`, tensor: attnOut });
            } else {
                attnOut.dispose();
            }

            return this.getMLPOutput(residual1, attrs);
        });
    }

    dispose() {
        this.ln1.dispose();
        this.attn.dispose();
        this.ln2.dispose();
        this.mlp.dispose();
    }
}
