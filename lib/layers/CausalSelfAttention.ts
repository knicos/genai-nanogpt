import { attentionMask } from '../ops/attentionMask';
import BaseLayer, { GPTLayerConfig } from './BaseLayer';
import { qkv } from '../ops/qkv';
import { rope } from '../ops/rope';
import { appendCache } from '@base/ops/appendCache';
import {
    customGrad,
    dropout,
    engine,
    fill,
    grads,
    keep,
    linalg,
    matMul,
    ones,
    randomNormal,
    reshape,
    Tensor,
    tidy,
    variable,
    Variable,
    where,
    zeros,
} from '@tensorflow/tfjs-core';
import { fusedSoftmax } from '@base/ops/fusedSoftmax';
import { dot } from '@tensorflow/tfjs-layers/dist/backend/tfjs_backend';

export type KVCache = {
    k: Tensor; // [B, nHead, T_cache, headDim]
    v: Tensor; // [B, nHead, T_cache, headDim]
    length: number;
    cumulativeLength: number;
};

// Multi-head self-attention implementation
export default class CausalSelfAttention extends BaseLayer {
    private cAttn: Variable | null = null;
    private cProj: Variable | null = null;
    private bias: Tensor;
    private maskInf: Tensor;
    private divisor: number;
    private index: number;
    private _trainable: boolean = true;
    private units: number;
    private projUnits: number;

    constructor(index: number, config: GPTLayerConfig) {
        super(config);
        this.index = index;
        this.units = config.gpt.nEmbed * 3;
        this.projUnits = config.gpt.nEmbed;

        // Causal mask to ensure that attention is only applied to the left in the input sequence
        this.bias = linalg.bandPart(ones([config.gpt.blockSize, config.gpt.blockSize]), -1, 0).cast('bool');
        this.divisor = 1 / Math.sqrt(config.gpt.nEmbed / config.gpt.nHead); // Scaling factor for attention scores
        const zero = zeros([config.gpt.blockSize, config.gpt.blockSize]);
        // It must be negative infinity for softmax to ignore these positions
        // Using any other number results in small but non-zero attention weights
        // Which leaks information from the future
        const negInf = fill([config.gpt.blockSize, config.gpt.blockSize], Number.NEGATIVE_INFINITY);
        this.maskInf = where(this.bias as Tensor, zero, negInf);
    }

    private build() {
        if (this.cAttn === null) {
            this.cAttn = variable(
                randomNormal([this.config.gpt.nEmbed, this.units], 0, 0.02),
                true
                //`block_${this.index}_attn_cAttn_kernel`
            );
        }
        if (this.cProj === null) {
            this.cProj = variable(
                randomNormal([this.projUnits, this.config.gpt.nEmbed], 0, 0.02),
                true
                //`block_${this.index}_attn_cProj_kernel`
            );
        }
    }

    get variables(): Variable[] {
        if (this.cAttn === null) {
            throw new Error('Layer not built yet');
        }
        return [this.cAttn, this.cProj!];
    }

    get trainable(): boolean {
        return this._trainable;
    }

    set trainable(value: boolean) {
        this._trainable = value;
        if (this.cAttn) this.cAttn.trainable = value;
        if (this.cProj) this.cProj.trainable = value;
    }

    saveWeights(map: Map<string, Tensor[]>) {
        map.set(`block_${this.index}_cAttn`, this.cAttn ? [this.cAttn.clone()] : []);
        map.set(`block_${this.index}_cProj`, this.cProj ? [this.cProj.clone()] : []);
    }

    loadWeights(weights: Map<string, Tensor[]>): void {
        const attnWeight = weights.get(`block_${this.index}_cAttn`)?.[0];
        const projWeight = weights.get(`block_${this.index}_cProj`)?.[0];
        if (!attnWeight) throw new Error(`Weights for block_${this.index}_cAttn not found`);
        if (!projWeight) throw new Error(`Weights for block_${this.index}_cProj not found`);
        if (this.cAttn) {
            this.cAttn.assign(attnWeight);
        } else {
            this.cAttn = variable(attnWeight, true); //, `block_${this.index}_attn_cAttn_kernel`);
        }
        if (this.cProj) {
            this.cProj.assign(projWeight);
        } else {
            this.cProj = variable(projWeight, true); //, `block_${this.index}_attn_cProj_kernel`);
        }
    }

    private getAttentionScores(q: Tensor, k: Tensor, training: boolean, seed: number): Tensor {
        const maskedAtt = attentionMask(q, k, this.divisor, this.maskInf);
        return fusedSoftmax(maskedAtt, training ? this.config.gpt.dropout : 0, seed);
    }

    // Attention with optional past. If pastLen > 0 and T_cur == 1, no mask needed.
    private getAttentionScoresWithPast(
        q: Tensor, // [B, nh, T_cur, hs]
        kTotal: Tensor, // [B, nh, T_total, hs] where T_total=pastLen+T_cur
        pastLen: number
    ): Tensor {
        const att = attentionMask(q, kTotal, this.divisor, undefined, pastLen);
        return fusedSoftmax(att, 0, 0);
    }

    private getQKV(x: Tensor): [Tensor, Tensor, Tensor] {
        return qkv(x, this.cAttn!, this.config.gpt.nHead) as [Tensor, Tensor, Tensor];
    }

    private getOutputProjection(x: Tensor): Tensor {
        const B = x.shape[0]!; // batch size
        const T = x.shape[2]!; // sequence length
        const C = this.config.gpt.nEmbed; // embedding dimensionality

        // Re-assemble all head outputs side by side
        const yTransposed = x.transpose([0, 2, 1, 3]); // (B, T, nh, hs)
        const yReshaped = reshape(yTransposed, [B, T, C]); // (B, T, C)

        // Output projection
        // This dot is used by dense layers so it should be optimized
        const output = dot(yReshaped, this.cProj!);
        return output;
    }

    private updateCache(kNew: Tensor, vNew: Tensor, skip: boolean, pastKV?: KVCache): KVCache {
        const maxCtx = this.config.gpt.blockSize;
        const Tcur = kNew.shape[2]!;
        const pastLen = pastKV?.length || 0;

        // Append and trim cache to max context size
        const kTotal = skip ? kNew : appendCache(kNew, maxCtx, pastLen, pastKV?.k);
        if (!skip) {
            kNew.dispose();
            pastKV?.k.dispose();
        }

        const vTotal = skip ? vNew : appendCache(vNew, maxCtx, pastLen, pastKV?.v);
        if (!skip) {
            vNew.dispose();
            pastKV?.v.dispose();
        }

        const presentKV: KVCache = {
            k: keep(kTotal),
            v: keep(vTotal),
            length: Math.min(pastLen + Tcur, maxCtx),
            cumulativeLength: pastKV ? pastKV.cumulativeLength + Tcur : Tcur,
        };
        return presentKV;
    }

    private forward(
        x: Tensor,
        training = false,
        seed: number,
        includeAttention = false,
        pastKV?: KVCache
    ): { output: Tensor; attention?: Tensor; presentKV?: KVCache } {
        return tidy(() => {
            this.startMemory();
            const [qI, kNewI, vNew] = this.getQKV(x); // q: [B,nh,T_cur,hs], kNew/vNew: [B,nh,T_cur,hs]

            // Apply RoPE to current chunk before concatenating with past
            // The rope operator ensures the cache is large enough
            const pastLenInitial = pastKV ? pastKV.cumulativeLength : 0;
            const ropeCache = this.config.layerConfig.ropeCache;
            const q = ropeCache ? rope(qI, ropeCache, pastLenInitial) : qI;
            const kNew = ropeCache ? rope(kNewI, ropeCache, pastLenInitial) : kNewI;

            if (ropeCache) {
                qI.dispose();
                kNewI.dispose();
            }

            const pastLen = pastKV ? pastKV.length : 0;
            const presentKV = this.updateCache(kNew, vNew, training, pastKV);
            const kTotal = presentKV.k;
            const vTotal = presentKV.v;

            // Attention scores: mask for full forward or multi-token chunk; skip for single-token incremental
            let attScores: Tensor;
            if (pastLen > 0) {
                attScores = this.getAttentionScoresWithPast(q, kTotal, pastLen);
            } else {
                // No past: regular causal mask over a square (training/full forward)
                attScores = this.getAttentionScores(q, kTotal, training, seed);
            }
            q.dispose();
            if (training) {
                kTotal.dispose();
            }

            // Attention applied to values
            const y = matMul(attScores, vTotal); // (B, nh, T_cur, hs)
            if (!includeAttention) {
                attScores.dispose();
            }
            if (training) {
                vTotal.dispose();
            }

            const output = this.getOutputProjection(y); // (B, T_cur, C)
            y.dispose();

            const attention = includeAttention ? attScores.mean(1) : undefined;
            this.endMemory(`CausalSelfAttention`);
            return { output, attention, presentKV: training ? undefined : presentKV };
        });
    }

    call(
        x: Tensor,
        training = false,
        includeAttention = false,
        pastKV?: KVCache
    ): { output: Tensor; attention?: Tensor; presentKV?: KVCache } {
        if (pastKV && !this.config.gpt.useRope) {
            throw new Error('Cannot use pastKV without RoPE enabled');
        }
        if (training && pastKV) {
            throw new Error('Cannot use pastKV during training');
        }
        if (x.shape.length !== 3) {
            throw new Error(`Input tensor must be rank 3 [B, T, C], got shape ${x.shape}`);
        }
        if (x.shape[2] !== this.config.gpt.nEmbed) {
            throw new Error(`Input tensor last dimension must be ${this.config.gpt.nEmbed}, got ${x.shape[2]}`);
        }

        this.build();

        const seed = Math.random() * 1e9;

        if (training && this.config.layerConfig.checkpointAttention) {
            const cpAttention = customGrad(
                // @ts-expect-error Invalid params
                (norm: Tensor, attnVar: Tensor, projVar: Tensor, save: (tensors: Tensor[]) => void) => {
                    const attnOut = this.forward(norm, true, seed);

                    save([norm]);
                    const gradFunc = (dy: Tensor, saved: Tensor[]) => {
                        const [normSaved] = saved;

                        // Hack to allow nested grads calls
                        const savedTape = engine().state.activeTape;
                        engine().state.activeTape = [];

                        // Recompute forward pass
                        // We need to pass attnVar and projVar to keep them in scope for the backward pass
                        const g = grads((n: Tensor, attnVar: Tensor, projVar: Tensor) => {
                            void attnVar;
                            void projVar;
                            const attnOut = this.forward(n, true, seed);
                            return attnOut.output;
                        })([normSaved, attnVar, projVar], dy);

                        // Restore tape
                        engine().state.activeTape = savedTape;

                        return g;
                    };
                    return { value: attnOut.output, gradFunc };
                }
            );

            const attnOut = cpAttention(x, this.cAttn!, this.cProj!);

            // Dropout after checkpointing
            if (this.config.gpt.dropout > 0) {
                const finalOutput = dropout(attnOut, this.config.gpt.dropout);
                attnOut.dispose();
                return { output: finalOutput };
            } else {
                return { output: attnOut };
            }
        } else {
            const output = this.forward(x, training, seed, includeAttention, pastKV);
            // Dropout after checkpointing
            if (this.config.gpt.dropout > 0) {
                const finalOutput = dropout(output.output, this.config.gpt.dropout);
                output.output.dispose();
                return { output: finalOutput, attention: output.attention, presentKV: output.presentKV };
            } else {
                return output;
            }
        }
    }

    dispose() {
        this.cAttn?.dispose();
        this.cProj?.dispose();
        this.bias.dispose();
        this.maskInf.dispose();
    }
}
