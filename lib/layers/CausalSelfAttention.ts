import { GPTConfig } from '../config';
import RoPECache from './RoPECache';
import { attentionMask } from '../ops/attentionMask';
import BaseLayer from './BaseLayer';
import { qkv } from '../ops/qkv';
import { rope } from '../ops/rope';
import { appendCache } from '@base/ops/appendCache';
import {
    fill,
    keep,
    linalg,
    matMul,
    ones,
    randomNormal,
    reshape,
    softmax,
    Tensor,
    tidy,
    variable,
    Variable,
    where,
    zeros,
} from '@tensorflow/tfjs-core';
import { initializers, layers } from '@tensorflow/tfjs-layers';

export type KVCache = {
    k: Tensor; // [B, nHead, T_cache, headDim]
    v: Tensor; // [B, nHead, T_cache, headDim]
    length: number;
    cumulativeLength: number;
};

// Multi-head self-attention implementation
export default class CausalSelfAttention extends BaseLayer {
    private config: GPTConfig;
    private cAttn: Variable | null = null;
    private cProj: layers.Layer;
    private attnDropout: layers.Layer;
    private residDropout: layers.Layer;
    private bias: Tensor;
    private maskInf: Tensor;
    private divisor: number;
    private index: number;
    private _trainable: boolean = true;
    private units: number;

    constructor(index: number, config: GPTConfig, private readonly ropeCache?: RoPECache) {
        super();
        this.config = config;
        this.index = index;
        this.units = config.nEmbed * 3;

        // Key, query, value projections for all heads, but in a batch
        /*this.cAttn = this.tf.layers.dense({
            units: 3 * config.nEmbed,
            useBias: config.biasInLinear,
            name: `block_${index}_attn_cAttn`,
            kernelInitializer: this.tf.initializers.randomNormal({
                mean: 0.0,
                stddev: 0.02,
            }),
            biasInitializer: 'zeros',
        });*/

        // Output projection
        this.cProj = layers.dense({
            units: config.nEmbed,
            useBias: config.biasInLinear,
            name: `block_${index}_attn_cProj`,
            kernelInitializer: initializers.randomNormal({
                mean: 0.0,
                stddev: 0.02 / Math.sqrt(2 * config.nLayer),
            }),
            biasInitializer: 'zeros',
        });

        // Dropout layers
        this.attnDropout = layers.dropout({ rate: config.dropout });
        this.residDropout = layers.dropout({ rate: config.dropout });

        // Causal mask to ensure that attention is only applied to the left in the input sequence
        this.bias = linalg.bandPart(ones([config.blockSize, config.blockSize]), -1, 0).cast('bool');
        this.divisor = 1 / Math.sqrt(config.nEmbed / config.nHead); // Scaling factor for attention scores
        const zero = zeros([config.blockSize, config.blockSize]);
        // It must be negative infinity for softmax to ignore these positions
        const negInf = fill([config.blockSize, config.blockSize], Number.NEGATIVE_INFINITY);
        this.maskInf = where(this.bias as Tensor, zero, negInf);
    }

    private build() {
        if (this.cAttn === null) {
            this.cAttn = variable(
                randomNormal([this.config.nEmbed, this.units], 0, 0.02),
                true
                //`block_${this.index}_attn_cAttn_kernel`
            );
        }
    }

    get variables(): Variable[] {
        if (this.cAttn === null) {
            throw new Error('Layer not built yet');
        }
        return [this.cAttn, ...this.cProj.trainableWeights.map((v) => v.read() as Variable)];
    }

    get trainable(): boolean {
        return this._trainable;
    }

    set trainable(value: boolean) {
        this._trainable = value;
        if (this.cAttn) this.cAttn.trainable = value;
        this.cProj.trainable = value;
    }

    saveWeights(map: Map<string, Tensor[]>) {
        map.set(`block_${this.index}_cAttn`, this.cAttn ? [this.cAttn.clone()] : []);
        map.set(`block_${this.index}_cProj`, this.cProj.getWeights());
    }

    loadWeights(weights: Map<string, Tensor[]>): void {
        const attnWeight = weights.get(`block_${this.index}_cAttn`)?.[0];
        if (!attnWeight) throw new Error(`Weights for block_${this.index}_cAttn not found`);
        if (this.cAttn) {
            this.cAttn.assign(attnWeight);
        } else {
            this.cAttn = variable(attnWeight, true); //, `block_${this.index}_attn_cAttn_kernel`);
        }
        this.cProj.setWeights(weights.get(`block_${this.index}_cProj`) || []);
    }

    private getAttentionScores(q: Tensor, k: Tensor, training: boolean): Tensor {
        const maskedAtt = attentionMask(q, k, this.maskInf, this.divisor);
        const attSoftmax = softmax(maskedAtt, -1); // (B, nh, T, T)
        return this.attnDropout.apply(attSoftmax, { training }) as Tensor;
    }

    // Attention with optional past. If pastLen > 0 and T_cur == 1, no mask needed.
    private getAttentionScoresWithPast(
        q: Tensor, // [B, nh, T_cur, hs]
        kTotal: Tensor, // [B, nh, T_total, hs] where T_total=pastLen+T_cur
        training: boolean,
        pastLen: number
    ): Tensor {
        const Tcur = q.shape[2]!;

        const attUnscaled = matMul(q, kTotal, false, true); // (B, nh, T_cur, T_total)
        let att = attUnscaled.mul(this.divisor);

        if (Tcur > 1 && pastLen > 0) {
            throw new Error('Cannot use past with T_cur > 1'); // This should not happen
        }

        // Mask only needed if there is more than one token in the current chunk
        if (Tcur > 1) {
            const mask = this.maskInf.slice([0, 0], [Tcur, Tcur]).expandDims(0).expandDims(0); // (1,1,T_cur,T_cur)
            att = att.add(mask);
        }
        const attSoftmax = softmax(att, -1);
        return this.attnDropout.apply(attSoftmax, { training }) as Tensor;
    }

    private getQKV(x: Tensor): [Tensor, Tensor, Tensor] {
        return qkv(x, this.cAttn!, this.config.nHead) as [Tensor, Tensor, Tensor];
    }

    private getOutputProjection(x: Tensor, training: boolean): Tensor {
        const B = x.shape[0]!; // batch size
        const T = x.shape[2]!; // sequence length
        const C = this.config.nEmbed; // embedding dimensionality

        // Re-assemble all head outputs side by side
        const yTransposed = x.transpose([0, 2, 1, 3]); // (B, T, nh, hs)
        const yReshaped = reshape(yTransposed, [B, T, C]); // (B, T, C)

        // Output projection
        const output = this.cProj.apply(yReshaped) as Tensor;
        const finalOutput = this.residDropout.apply(output, { training }) as Tensor;
        return finalOutput;
    }

    private updateCache(kNew: Tensor, vNew: Tensor, pastKV?: KVCache): KVCache {
        const maxCtx = this.config.blockSize;
        const Tcur = kNew.shape[2]!;
        const pastLen = Math.min(pastKV?.length || 0, maxCtx - Tcur);

        const kTotal = pastKV ? appendCache(pastKV.k, kNew, maxCtx) : kNew;
        const vTotal = pastKV ? appendCache(pastKV.v, vNew, maxCtx) : vNew;

        // Handled in the NanoGPTModel class
        /*if (pastKV) {
            pastKV.k.dispose();
            pastKV.v.dispose();
        }*/

        const presentKV: KVCache = {
            k: keep(kTotal),
            v: keep(vTotal),
            length: pastLen + Tcur,
            cumulativeLength: pastKV ? pastKV.cumulativeLength + Tcur : Tcur,
        };
        return presentKV;
    }

    // Added optional KV cache support (pastKV). Returns presentKV for chaining.
    call(
        x: Tensor,
        training = false,
        includeAttention = false,
        pastKV?: KVCache
    ): { output: Tensor; attention?: Tensor; presentKV?: KVCache } {
        if (pastKV && !this.config.useRope) {
            throw new Error('Cannot use pastKV without RoPE enabled');
        }

        this.build();

        return tidy(() => {
            this.startMemory();
            const [qI, kNewI, vNew] = this.getQKV(x); // q: [B,nh,T_cur,hs], kNew/vNew: [B,nh,T_cur,hs]

            // Apply RoPE to current chunk before concatenating with past
            const pastLenInitial = pastKV ? pastKV.cumulativeLength : 0;
            //const [q, kNew] = this.ropeCache ? this.ropeCache.applyRoPE(qI, kNewI, pastLenInitial) : [qI, kNewI];
            const q = this.ropeCache ? rope(qI, this.ropeCache, pastLenInitial) : qI;
            const kNew = this.ropeCache ? rope(kNewI, this.ropeCache, pastLenInitial) : kNewI;

            if (this.ropeCache) {
                qI.dispose();
                kNewI.dispose();
            }

            const pastLen = pastKV ? pastKV.length : 0;
            const presentKV = this.updateCache(kNew, vNew, pastKV);
            const kTotal = presentKV.k;
            const vTotal = presentKV.v;

            if (pastKV) {
                kNew.dispose();
                vNew.dispose();
            }

            // Attention scores: mask for full forward or multi-token chunk; skip for single-token incremental
            let attScores: Tensor;
            if (pastLen > 0) {
                attScores = this.getAttentionScoresWithPast(q, kTotal, training, pastLen);
            } else {
                // No past: regular causal mask over a square (training/full forward)
                attScores = this.getAttentionScores(q, kTotal, training);
            }

            // Attention applied to values
            const y = matMul(attScores, vTotal); // (B, nh, T_cur, hs)

            const output = this.getOutputProjection(y, training); // (B, T_cur, C)

            const attention = includeAttention ? attScores.mean(1) : undefined;
            this.endMemory(`CausalSelfAttention`);
            return { output, attention, presentKV };
        });
    }

    dispose() {
        this.cAttn?.dispose();
        this.cProj.dispose();
        this.attnDropout.dispose();
        this.residDropout.dispose();
        this.bias.dispose();
        this.maskInf.dispose();
    }
}
