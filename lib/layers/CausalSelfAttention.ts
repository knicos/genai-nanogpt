import type TF from '@tensorflow/tfjs';
import { GPTConfig } from '../config';
import RoPECache from './RoPECache';
import { attentionMask } from '@base/ops/attentionMask';

export type KVCache = {
    k: TF.Tensor; // [B, nHead, T_cache, headDim]
    v: TF.Tensor; // [B, nHead, T_cache, headDim]
    length: number;
    cumulativeLength: number;
};

// Multi-head self-attention implementation
export default class CausalSelfAttention {
    private config: GPTConfig;
    private cAttn: TF.layers.Layer;
    private cProj: TF.layers.Layer;
    private attnDropout: TF.layers.Layer;
    private residDropout: TF.layers.Layer;
    private bias: TF.Tensor;
    private maskInf: TF.Tensor;
    private tf: typeof TF;
    private divisor: number;
    private index: number;
    private _trainable: boolean = true;

    constructor(tf: typeof TF, index: number, config: GPTConfig, private readonly ropeCache?: RoPECache) {
        this.config = config;
        this.tf = tf;
        this.index = index;

        // Key, query, value projections for all heads, but in a batch
        this.cAttn = this.tf.layers.dense({
            units: 3 * config.nEmbed,
            useBias: config.biasInLinear,
            name: `block_${index}_attn_cAttn`,
            kernelInitializer: this.tf.initializers.randomNormal({
                mean: 0.0,
                stddev: 0.02,
            }),
            biasInitializer: 'zeros',
        });

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

    get variables(): TF.Variable[] {
        return [
            ...this.cAttn.trainableWeights.map((v) => v.read() as TF.Variable),
            ...this.cProj.trainableWeights.map((v) => v.read() as TF.Variable),
        ];
    }

    get trainable(): boolean {
        return this._trainable;
    }

    set trainable(value: boolean) {
        this._trainable = value;
        this.cAttn.trainable = value;
        this.cProj.trainable = value;
    }

    saveWeights(map: Map<string, TF.Tensor[]>) {
        map.set(`block_${this.index}_cAttn`, this.cAttn.getWeights());
        map.set(`block_${this.index}_cProj`, this.cProj.getWeights());
    }

    loadWeights(weights: Map<string, TF.Tensor[]>): void {
        this.cAttn.setWeights(weights.get(`block_${this.index}_cAttn`) || []);
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
        const [B, T, C] = x.shape; // batch size, sequence length, embedding dimensionality

        // Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        const qkv = this.cAttn.apply(x) as TF.Tensor; // (B, T, 3*C)
        //x.dispose();
        const [q, k, v] = this.tf.split(qkv, 3, -1); // Each is (B, T, C)
        qkv.dispose();

        // Reshape for multi-head attention
        const headDim = C / this.config.nHead;

        const qReshaped = this.tf.reshape(q, [B, T, this.config.nHead, headDim]);
        q.dispose();
        const qT = qReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
        qReshaped.dispose();

        const kReshaped = this.tf.reshape(k, [B, T, this.config.nHead, headDim]);
        k.dispose();
        const kT = kReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
        kReshaped.dispose();

        const vReshaped = this.tf.reshape(v, [B, T, this.config.nHead, headDim]);
        v.dispose();
        const vT = vReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
        vReshaped.dispose();
        return [qT, kT, vT];
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
        return this.tf.tidy(() => {
            const [qI, kNewI, vNew] = this.getQKV(x); // q: [B,nh,T_cur,hs], kNew/vNew: [B,nh,T_cur,hs]
            const Tcur = qI.shape[2]!;
            const maxCtx = this.config.blockSize;

            // Apply RoPE to current chunk before concatenating with past
            const pastLenInitial = pastKV ? pastKV.cumulativeLength : 0;
            const [q, kNew] = this.ropeCache ? this.ropeCache.applyRoPE(qI, kNewI, pastLenInitial) : [qI, kNewI];

            let kTotal = kNew;
            let vTotal = vNew;
            let pastLen = 0;

            if (pastKV) {
                pastLen = pastKV.length;
                kTotal = this.tf.concat([pastKV.k, kNew], 2); // [B,nh,T_total,hs]
                vTotal = this.tf.concat([pastKV.v, vNew], 2); // [B,nh,T_total,hs]
            }

            // Clamp to sliding window [last maxCtx tokens]
            const Ttotal = kTotal.shape[2]!;
            if (Ttotal > maxCtx) {
                const start = Ttotal - maxCtx;
                const B = kTotal.shape[0]!;
                const H = kTotal.shape[1]!;
                const HS = kTotal.shape[3]!;
                kTotal = kTotal.slice([0, 0, start, 0], [B, H, maxCtx, HS]);
                vTotal = vTotal.slice([0, 0, start, 0], [B, H, maxCtx, HS]);

                // Effective past after clamping
                const clampedTotal = maxCtx;
                pastLen = clampedTotal - Tcur;
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

            // Prepare present cache; keep so caller can reuse outside tidy
            const presentKV: KVCache = {
                k: this.tf.keep(kTotal),
                v: this.tf.keep(vTotal),
                length: pastLen + Tcur,
                cumulativeLength: pastKV ? pastKV.cumulativeLength + Tcur : Tcur,
            };

            return { output, attention: includeAttention ? attScores.mean(1) : undefined, presentKV };
        });
    }

    dispose() {
        this.cAttn.dispose();
        this.cProj.dispose();
        this.attnDropout.dispose();
        this.residDropout.dispose();
        this.bias.dispose();
        this.maskInf.dispose();
    }
}
