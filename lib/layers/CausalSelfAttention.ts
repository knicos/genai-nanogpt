import type TF from '@tensorflow/tfjs';
import { GPTConfig } from '../config';
import RoPECache from './RoPECache';
import { attentionMask } from '../ops/attentionMask';
import BaseLayer from './BaseLayer';
import { qkv } from '../ops/qkv';
import { rope } from '../ops/rope';
import { appendCache } from '@base/ops/appendCache';

export type KVCache = {
    k: TF.Tensor; // [B, nHead, T_cache, headDim]
    v: TF.Tensor; // [B, nHead, T_cache, headDim]
    length: number;
    cumulativeLength: number;
};

// Multi-head self-attention implementation
export default class CausalSelfAttention extends BaseLayer {
    private config: GPTConfig;
    private cAttn: TF.Variable | null = null;
    private cProj: TF.layers.Layer;
    private attnDropout: TF.layers.Layer;
    private residDropout: TF.layers.Layer;
    private bias: TF.Tensor;
    private maskInf: TF.Tensor;
    private tf: typeof TF;
    private divisor: number;
    private index: number;
    private _trainable: boolean = true;
    private units: number;

    constructor(tf: typeof TF, index: number, config: GPTConfig, private readonly ropeCache?: RoPECache) {
        super();
        this.config = config;
        this.tf = tf;
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
        this.cProj = this.tf.layers.dense({
            units: config.nEmbed,
            useBias: config.biasInLinear,
            name: `block_${index}_attn_cProj`,
            kernelInitializer: this.tf.initializers.randomNormal({
                mean: 0.0,
                stddev: 0.02 / Math.sqrt(2 * config.nLayer),
            }),
            biasInitializer: 'zeros',
        });

        // Dropout layers
        this.attnDropout = this.tf.layers.dropout({ rate: config.dropout });
        this.residDropout = this.tf.layers.dropout({ rate: config.dropout });

        // Causal mask to ensure that attention is only applied to the left in the input sequence
        this.bias = this.tf.linalg.bandPart(this.tf.ones([config.blockSize, config.blockSize]), -1, 0).cast('bool');
        this.divisor = 1 / Math.sqrt(config.nEmbed / config.nHead); // Scaling factor for attention scores
        const zeros = this.tf.zeros([config.blockSize, config.blockSize]);
        // It must be negative infinity for softmax to ignore these positions
        const negInf = this.tf.fill([config.blockSize, config.blockSize], Number.NEGATIVE_INFINITY);
        this.maskInf = this.tf.where(this.bias as TF.Tensor, zeros, negInf);
    }

    private build() {
        if (this.cAttn === null) {
            this.cAttn = this.tf.variable(
                this.tf.randomNormal([this.config.nEmbed, this.units], 0, 0.02),
                true
                //`block_${this.index}_attn_cAttn_kernel`
            );
        }
    }

    get variables(): TF.Variable[] {
        if (this.cAttn === null) {
            throw new Error('Layer not built yet');
        }
        return [this.cAttn, ...this.cProj.trainableWeights.map((v) => v.read() as TF.Variable)];
    }

    get trainable(): boolean {
        return this._trainable;
    }

    set trainable(value: boolean) {
        this._trainable = value;
        if (this.cAttn) this.cAttn.trainable = value;
        this.cProj.trainable = value;
    }

    saveWeights(map: Map<string, TF.Tensor[]>) {
        map.set(`block_${this.index}_cAttn`, this.cAttn ? [this.cAttn.clone()] : []);
        map.set(`block_${this.index}_cProj`, this.cProj.getWeights());
    }

    loadWeights(weights: Map<string, TF.Tensor[]>): void {
        const attnWeight = weights.get(`block_${this.index}_cAttn`)?.[0];
        if (!attnWeight) throw new Error(`Weights for block_${this.index}_cAttn not found`);
        if (this.cAttn) {
            this.cAttn.assign(attnWeight);
        } else {
            this.cAttn = this.tf.variable(attnWeight, true); //, `block_${this.index}_attn_cAttn_kernel`);
        }
        this.cProj.setWeights(weights.get(`block_${this.index}_cProj`) || []);
    }

    private getAttentionScores(q: TF.Tensor, k: TF.Tensor, training: boolean): TF.Tensor {
        const maskedAtt = attentionMask(q, k, this.maskInf, this.divisor);
        const attSoftmax = this.tf.softmax(maskedAtt, -1); // (B, nh, T, T)
        return this.attnDropout.apply(attSoftmax, { training }) as TF.Tensor;
    }

    // Attention with optional past. If pastLen > 0 and T_cur == 1, no mask needed.
    private getAttentionScoresWithPast(
        q: TF.Tensor, // [B, nh, T_cur, hs]
        kTotal: TF.Tensor, // [B, nh, T_total, hs] where T_total=pastLen+T_cur
        training: boolean,
        pastLen: number
    ): TF.Tensor {
        const Tcur = q.shape[2]!;

        const attUnscaled = this.tf.matMul(q, kTotal, false, true); // (B, nh, T_cur, T_total)
        let att = attUnscaled.mul(this.divisor);

        if (Tcur > 1 && pastLen > 0) {
            throw new Error('Cannot use past with T_cur > 1'); // This should not happen
        }

        // Mask only needed if there is more than one token in the current chunk
        if (Tcur > 1) {
            const mask = this.maskInf.slice([0, 0], [Tcur, Tcur]).expandDims(0).expandDims(0); // (1,1,T_cur,T_cur)
            att = att.add(mask);
        }
        const attSoftmax = this.tf.softmax(att, -1);
        return this.attnDropout.apply(attSoftmax, { training }) as TF.Tensor;
    }

    private getQKV(x: TF.Tensor): [TF.Tensor, TF.Tensor, TF.Tensor] {
        return qkv(x, this.cAttn!, this.config.nHead) as [TF.Tensor, TF.Tensor, TF.Tensor];
    }

    private getOutputProjection(x: TF.Tensor, training: boolean): TF.Tensor {
        const B = x.shape[0]!; // batch size
        const T = x.shape[2]!; // sequence length
        const C = this.config.nEmbed; // embedding dimensionality

        // Re-assemble all head outputs side by side
        const yTransposed = x.transpose([0, 2, 1, 3]); // (B, T, nh, hs)
        const yReshaped = this.tf.reshape(yTransposed, [B, T, C]); // (B, T, C)

        // Output projection
        const output = this.cProj.apply(yReshaped) as TF.Tensor;
        const finalOutput = this.residDropout.apply(output, { training }) as TF.Tensor;
        return finalOutput;
    }

    private updateCache(kNew: TF.Tensor, vNew: TF.Tensor, pastKV?: KVCache): KVCache {
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
            k: this.tf.keep(kTotal),
            v: this.tf.keep(vTotal),
            length: pastLen + Tcur,
            cumulativeLength: pastKV ? pastKV.cumulativeLength + Tcur : Tcur,
        };
        return presentKV;
    }

    // Added optional KV cache support (pastKV). Returns presentKV for chaining.
    call(
        x: TF.Tensor,
        training = false,
        includeAttention = false,
        pastKV?: KVCache
    ): { output: TF.Tensor; attention?: TF.Tensor; presentKV?: KVCache } {
        if (pastKV && !this.config.useRope) {
            throw new Error('Cannot use pastKV without RoPE enabled');
        }

        this.build();

        return this.tf.tidy(() => {
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
            let attScores: TF.Tensor;
            if (pastLen > 0) {
                attScores = this.getAttentionScoresWithPast(q, kTotal, training, pastLen);
            } else {
                // No past: regular causal mask over a square (training/full forward)
                attScores = this.getAttentionScores(q, kTotal, training);
            }

            // Attention applied to values
            const y = this.tf.matMul(attScores, vTotal); // (B, nh, T_cur, hs)

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
