import CausalSelfAttention, { AttentionScores, CausalSelfAttentionConfig, KVCache } from './CausalSelfAttention';
import MLP, { MLPConfig } from './MLP';
import RMSNorm, { RMSNormConfig } from './RMSNorm';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { keep, Tensor, tidy } from '@tensorflow/tfjs-core';
import { GPTConfig } from '@base/models/config';
import { add16 } from '@base/ops/add16';

interface BlockAttributes extends ForwardAttributes {
    pastKV?: KVCache;
    seed?: number;
    attentionScores?: AttentionScores;
}

export type TransformerBlockConfig = MLPConfig & RMSNormConfig & CausalSelfAttentionConfig;

// Transformer block
export default class Block extends BaseLayer<BlockAttributes> {
    private ln1: RMSNorm;
    private attn: CausalSelfAttention;
    private ln2: RMSNorm;
    private mlp: MLP;
    private index: number;
    public skipped = false;
    private blockConfig: TransformerBlockConfig;

    constructor(index: number, config: GPTConfig, blockConfig: TransformerBlockConfig, parent?: BaseLayer) {
        super(config, parent);
        this.index = index;
        this.blockConfig = blockConfig;

        this.ln1 = new RMSNorm(config, this.blockConfig, `block_${this.index}_rms1`, this);

        this.attn = new CausalSelfAttention(this.index, config, this.blockConfig, this);

        this.ln2 = new RMSNorm(config, this.blockConfig, `block_${this.index}_rms2`, this);
        this.mlp = new MLP(this.index, config, this.blockConfig, this);
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
        const residual = add16(x, mlpOut);
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
            const residual1 = add16(x, attnOut);
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
