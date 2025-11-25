import { attentionMask } from '../ops/attentionMask';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { qkv } from '../ops/qkv';
import { rope } from '../ops/rope';
import { appendCache } from '@base/ops/appendCache';
import { dropout, keep, matMul, randomNormal, reshape, Tensor, tidy, variable } from '@tensorflow/tfjs-core';
import { fusedSoftmax } from '@base/ops/fusedSoftmax';
import { dot } from '@tensorflow/tfjs-layers/dist/backend/tfjs_backend';
import { GPTConfig } from '@base/models/config';

export type KVCache = {
    k?: Tensor; // [B, nHead, T_cache, headDim]
    v?: Tensor; // [B, nHead, T_cache, headDim]
    length: number;
    cumulativeLength: number;
};

export interface AttentionScores {
    meanOfHeads?: boolean; // If true, average attention weights over heads
    attentionOut?: Tensor[]; // [H, T, T] attention weights if requested
}

interface AttentionForwardAttributes extends ForwardAttributes {
    attentionScores?: AttentionScores;
    pastKV?: KVCache; // Optional past key/value cache for incremental decoding
    seed?: number; // Optional seed for dropout randomness
}

// Multi-head self-attention implementation
export default class CausalSelfAttention extends BaseLayer<AttentionForwardAttributes> {
    private divisor: number;
    private index: number;
    private units: number;
    private projUnits: number;
    private ATTN: string;
    private PROJ: string;

    constructor(index: number, config: GPTConfig, parent?: BaseLayer) {
        super(config, parent);
        this.index = index;
        this.units = config.nEmbed * 3;
        this.projUnits = config.nEmbed;

        this.ATTN = `block_${this.index}_cAttn`;
        this.PROJ = `block_${this.index}_cProj`;
        this.addVariable(this.ATTN);
        this.addVariable(this.PROJ);

        this.divisor = 1 / Math.sqrt(config.nEmbed / config.nHead); // Scaling factor for attention scores
    }

    protected override build() {
        if (this.hasVariable(this.ATTN) === false) {
            this.setVariable(
                this.ATTN,
                variable(
                    randomNormal([this.config.nEmbed, this.units], 0, 0.02),
                    true,
                    `block_${this.index}_attn_cAttn_kernel`
                )
            );
        }
        if (this.hasVariable(this.PROJ) === false) {
            this.setVariable(
                this.PROJ,
                variable(
                    randomNormal([this.projUnits, this.config.nEmbed], 0, 0.02),
                    true,
                    `block_${this.index}_attn_cProj_kernel`
                )
            );
        }
    }

    private getAttentionScores(q: Tensor, k: Tensor, training: boolean, seed: number): Tensor {
        const maskedAtt = attentionMask(q, k, this.divisor);
        const s = fusedSoftmax(maskedAtt, training ? this.config.dropout : 0, seed);
        maskedAtt.dispose();
        return s;
    }

    // Attention with optional past. If pastLen > 0 and T_cur == 1, no mask needed.
    private getAttentionScoresWithPast(
        q: Tensor, // [B, nh, T_cur, hs]
        kTotal: Tensor, // [B, nh, T_total, hs] where T_total=pastLen+T_cur
        pastLen: number
    ): Tensor {
        const att = attentionMask(q, kTotal, this.divisor, pastLen);
        const s = fusedSoftmax(att, 0, 0);
        att.dispose();
        return s;
    }

    private getQKV(x: Tensor): [Tensor, Tensor, Tensor] {
        return qkv(x, this.getVariable(this.ATTN), this.config.nHead) as [Tensor, Tensor, Tensor];
    }

    private getOutputProjection(x: Tensor): Tensor {
        const B = x.shape[0]!; // batch size
        const T = x.shape[2]!; // sequence length
        const C = this.config.nEmbed; // embedding dimensionality

        // Re-assemble all head outputs side by side
        const yTransposed = x.transpose([0, 2, 1, 3]); // (B, T, nh, hs)
        const yReshaped = reshape(yTransposed, [B, T, C]); // (B, T, C)

        // Output projection
        // This dot is used by dense layers so it should be optimized
        const output = dot(yReshaped, this.getVariable(this.PROJ));
        yReshaped.dispose();
        yTransposed.dispose();
        return output;
    }

    private updateCache(kNew: Tensor, vNew: Tensor, cache: KVCache) {
        const maxCtx = this.config.blockSize;
        const Tcur = kNew.shape[2]!;
        const pastLen = cache.length || 0;

        // Append and trim cache to max context size
        const kTotal = appendCache(kNew, maxCtx, pastLen, cache.k);
        kNew.dispose();
        if (cache.k) {
            cache.k.dispose();
        }

        const vTotal = appendCache(vNew, maxCtx, pastLen, cache.v);
        vNew.dispose();
        if (cache.v) {
            cache.v.dispose();
        }

        const length = Math.min(pastLen + Tcur, maxCtx);
        const cumulativeLength = cache.cumulativeLength + Tcur;
        cache.length = length;
        cache.cumulativeLength = cumulativeLength;
        cache.k = keep(kTotal);
        cache.v = keep(vTotal);
    }

    forward(attr: AttentionForwardAttributes, x: Tensor): Tensor {
        return tidy(() => {
            this.startMemory();
            const [qI, kNewI, vNew] = this.getQKV(x); // q: [B,nh,T_cur,hs], kNew/vNew: [B,nh,T_cur,hs]

            // Apply RoPE to current chunk before concatenating with past
            // The rope operator ensures the cache is large enough
            const pastLenInitial = attr.pastKV ? attr.pastKV.cumulativeLength : 0;
            const ropeCache = attr.ropeCache;
            const q = ropeCache ? rope(qI, ropeCache, pastLenInitial) : qI;
            const kNew = ropeCache ? rope(kNewI, ropeCache, pastLenInitial) : kNewI;

            if (ropeCache) {
                qI.dispose();
                kNewI.dispose();
            }

            const pastLen = attr.pastKV ? attr.pastKV.length : 0;
            if (attr.pastKV && !attr.training) {
                this.updateCache(kNew, vNew, attr.pastKV);
            }
            const kTotal = attr.pastKV?.k ? attr.pastKV.k : kNew;
            const vTotal = attr.pastKV?.v ? attr.pastKV.v : vNew;

            // Attention scores: mask for full forward or multi-token chunk; skip for single-token incremental
            let attScores: Tensor;
            if (pastLen > 0) {
                attScores = this.getAttentionScoresWithPast(q, kTotal, pastLen);
            } else {
                // No past: regular causal mask over a square (training/full forward)
                attScores = this.getAttentionScores(q, kTotal, attr.training, attr.seed || 0);
            }
            q.dispose();
            if (!attr.pastKV) {
                kTotal.dispose();
            }

            // Attention applied to values
            const y = matMul(attScores, vTotal); // (B, nh, T_cur, hs)

            const shouldOutputAttention =
                attr.attentionScores !== undefined && attr.attentionScores.attentionOut !== undefined;

            if (!shouldOutputAttention) {
                attScores.dispose();
            }
            if (!attr.pastKV) {
                vTotal.dispose();
            }

            const output = this.getOutputProjection(y); // (B, T_cur, C)
            y.dispose();

            // Optionally return attention scores for a head
            if (shouldOutputAttention && attr.attentionScores && attr.attentionScores.attentionOut !== undefined) {
                const H = attScores.shape[1]!;
                const T_cur = attScores.shape[2]!;

                attr.attentionScores.attentionOut?.push(
                    keep(attScores.slice([0, 0, 0, 0], [1, -1, -1, -1]).reshape([H, T_cur, -1]))
                );
            }
            this.endMemory(`CausalSelfAttention`);
            return output;
        });
    }

    protected override dropout(x: Tensor): Tensor {
        if (this.config.dropout > 0) {
            const finalOutput = dropout(x, this.config.dropout);
            x.dispose();
            return finalOutput;
        } else {
            return x;
        }
    }
}
