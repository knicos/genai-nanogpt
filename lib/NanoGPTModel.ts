import type TF from '@tensorflow/tfjs';
//import { PositionEmbedding } from './NLP/position_embedding';
import { defaultConfig, GPTConfig } from './config';
import Block from './layers/TransformerBlock';
import TiedEmbeddingOutputLayer from './layers/TiedEmbedding';
import { KVCache } from './layers/CausalSelfAttention';
import RoPECache from './layers/RoPECache';
import RMSNorm from './layers/RMSNorm';
import { estimateParameterCount } from './utilities/parameters';

export interface TrainingLogEntry {
    loss: number;
    valLoss?: number;
    step: number;
    time: number;
    example?: string;
    batchSize: number;
}

export interface GenerateOptions {
    temperature?: number;
    topK?: number;
    usePadding?: boolean;
    includeAttention?: boolean;
    includeProbabilities?: boolean;
}

// Main GPT model
export default class NanoGPT {
    public readonly config: GPTConfig;
    private wte: TiedEmbeddingOutputLayer; // Token embeddings
    private wpe?: TF.layers.Layer; // Position embeddings
    private drop: TF.layers.Layer; // Dropout
    private blocks: Block[];
    private lnF: RMSNorm; // Final layer norm
    private ropeCache?: RoPECache;
    public readonly tf: typeof TF;
    public log: TrainingLogEntry[] = []; // Training log

    constructor(tf: typeof TF, config: Partial<GPTConfig> = {}) {
        this.tf = tf;
        this.config = { ...defaultConfig, ...config };

        // Token embeddings
        this.wte = new TiedEmbeddingOutputLayer(tf, {
            vocabSize: this.config.vocabSize,
            embedDim: this.config.nEmbed,
            name: 'token_embedding',
        });

        if (this.config.useRope === false) {
            this.wpe = this.tf.layers.embedding({
                inputDim: this.config.blockSize,
                outputDim: this.config.nEmbed,
                name: 'positional_embedding',
                embeddingsInitializer: this.tf.initializers.randomNormal({ mean: 0.0, stddev: 0.02 }),
            });
        } else {
            this.ropeCache = new RoPECache(tf, this.config);
        }

        this.drop = this.tf.layers.dropout({ rate: this.config.dropout });

        // Transformer blocks
        this.blocks = [];
        for (let i = 0; i < this.config.nLayer; i++) {
            this.blocks.push(new Block(this.tf, i, this.config, this.ropeCache));
        }

        // Final layer norm
        this.lnF = new RMSNorm(tf, [this.config.nEmbed], 1e-8, `final_rms_norm`);
    }

    get variables(): TF.Variable[] {
        return [
            //...this.wpe.trainableWeights.map((v) => v.read() as TF.Variable),
            ...this.blocks.flatMap((b) => b.variables),
            ...this.lnF.trainableWeights.map((v) => v as TF.Variable),
            ...this.wte.variables,
        ];
    }

    public saveWeights(): Map<string, TF.Tensor[]> {
        const map = new Map<string, TF.Tensor[]>();
        map.set('token_embedding', this.wte.getWeights());
        if (this.wpe) map.set('positional_embedding', this.wpe.getWeights());
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].saveWeights(map);
        }
        map.set('final_rms_norm', this.lnF.getWeights());
        return map;
    }

    public loadWeights(weights: Map<string, TF.Tensor[]>): void {
        this.wte.setWeights(weights.get('token_embedding') || []);
        if (this.wpe) this.wpe.setWeights(weights.get('positional_embedding') || []);
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].loadWeights(weights);
        }
        this.lnF.setWeights(weights.get('final_rms_norm') || []);
    }

    private inputPhase(idx: TF.Tensor, pastLen: number, training: boolean = false): TF.Tensor {
        return this.tf.tidy(() => {
            //const [, seqLen] = idx.shape;
            //const maxCtx = this.config.blockSize;
            //const posStart = Math.min(pastLen, maxCtx - seqLen);

            const tokEmb = this.wte.embed(idx) as TF.Tensor; // (b, t, n_embd)

            if (this.config.useRope === false) {
                const [, seqLen] = idx.shape;
                const maxCtx = this.config.blockSize;
                // position_ids = (pastLen + arange(T)) % maxCtx    // stays in [0, blockSize)
                const range = this.tf.range(0, seqLen, 1, 'int32'); // (t,)
                const posIdx = this.tf.mod(
                    this.tf.add(range, this.tf.scalar(pastLen, 'int32')),
                    this.tf.scalar(maxCtx, 'int32')
                ) as TF.Tensor;
                const posEmb = this.wpe!.apply(posIdx) as TF.Tensor; // (b, t, n_embd)

                const embSum = tokEmb.add(posEmb);

                const out = this.drop.apply(embSum, { training }) as TF.Tensor;
                return out;
            } else {
                const out = this.drop.apply(tokEmb, { training }) as TF.Tensor;
                return out;
            }
        });
    }

    setSkipMask(mask: boolean[]): void {
        if (mask.length !== this.blocks.length) {
            throw new Error(`Mask length ${mask.length} does not match number of blocks ${this.blocks.length}`);
        }
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].skipped = mask[i];
        }
    }

    setTrainableMask(mask: boolean[]): void {
        if (mask.length !== this.blocks.length) {
            throw new Error(`Mask length ${mask.length} does not match number of blocks ${this.blocks.length}`);
        }
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].trainable = mask[i];
        }
    }

    set trainable(value: boolean) {
        for (const block of this.blocks) {
            block.trainable = value;
        }

        //this.wpe.trainable = value;
        this.lnF.trainable = value;
    }

    private validateInput(idx: TF.Tensor): void {
        if (idx.shape.length !== 2) {
            throw new Error(`Invalid input shape: expected [batch_size, sequence_length], got ${idx.shape}`);
        }
        if (idx.shape[1] > this.config.blockSize) {
            throw new Error(`Input sequence length ${idx.shape[1]} isn't block size ${this.config.blockSize}`);
        }
        if (idx.dtype !== 'int32') {
            throw new Error(`Input tensor must be of type int32, got ${idx.dtype}`);
        }
    }

    private calculateLoss(logits: TF.Tensor, targets: TF.Tensor): TF.Tensor {
        try {
            return this.tf.losses.softmaxCrossEntropy(targets, logits, this.tf.Reduction.MEAN);
        } catch (error) {
            console.error('Error computing loss:', error);
            throw new Error(`Loss computation failed: ${error}`);
        }
    }

    // Attention rollout per Abnar & Zuidema (2020)
    // Expects list of (B, T, T) attention matrices already averaged over heads.
    private computeAttentionRollout(attentions: TF.Tensor[]): TF.Tensor {
        return this.tf.tidy(() => {
            if (attentions.length === 0) {
                throw new Error('No attentions for rollout');
            }
            const [B, Q, K] = attentions[0].shape as number[];

            // Validate shapes are consistent
            for (const a of attentions) {
                const [b2, q2, k2] = a.shape as number[];
                if (b2 !== B || q2 !== Q || k2 !== K) {
                    throw new Error(
                        `Inconsistent attention shapes in rollout: expected [${B},${Q},${K}] got [${b2},${q2},${k2}]`
                    );
                }
            }

            if (Q === K) {
                // Full square attentions: standard rollout
                const eye = this.tf.eye(K, K).expandDims(0); // (1,K,K)
                let rollout = eye.tile([B, 1, 1]); // (B,K,K)
                for (const att of attentions) {
                    const a = att.add(eye);
                    const aNorm = a.div(a.sum(-1, true)); // (B,K,K)
                    rollout = aNorm.matMul(rollout); // (B,K,K)
                }
                return rollout;
            }

            if (Q === 1) {
                // Incremental (KV cache) attentions: only last row available per layer.
                // Compose a last-token rollout by combining layers' last-row distributions
                // with residual and renormalization.
                let rolloutRow: TF.Tensor | null = null; // (B,1,K)
                const lastIndex = this.tf.tensor1d([K - 1], 'int32');
                const lastOneHot = this.tf.oneHot(lastIndex, K).reshape([1, 1, K]).tile([B, 1, 1]); // (B,1,K)
                lastIndex.dispose();

                for (const att of attentions) {
                    let a = att.add(lastOneHot); // residual path on last position
                    a = a.div(a.sum(-1, true)); // row-normalize (B,1,K)
                    if (rolloutRow == null) {
                        rolloutRow = a;
                    } else {
                        rolloutRow = rolloutRow.mul(a);
                        rolloutRow = rolloutRow.div(rolloutRow.sum(-1, true)); // renormalize
                    }
                }
                return rolloutRow!;
            }

            throw new Error(`Unsupported attention shapes for rollout: [B=${B}, Q=${Q}, K=${K}]`);
        });
    }

    forward(
        idx: TF.Tensor,
        targets?: TF.Tensor,
        training = false,
        includeAttention = false,
        cache?: (KVCache | undefined)[]
    ): { logits: TF.Tensor; loss?: TF.Tensor; attention?: TF.Tensor } {
        this.validateInput(idx);

        return this.tf.tidy(() => {
            // Token and position embeddings
            const pastLen = cache?.[0]?.length ?? 0;
            let x = this.inputPhase(idx, pastLen, training);

            const perLayerAtt: TF.Tensor[] = [];

            if (cache && cache.length !== this.blocks.length) {
                console.error('Cache', cache);
                throw new Error(`Cache length ${cache.length} does not match number of blocks ${this.blocks.length}`);
            }

            // Transformer blocks
            for (let i = 0; i < this.blocks.length; i++) {
                const block = this.blocks[i];
                const {
                    output,
                    attention,
                    cache: newCache,
                } = block.call(x, training, includeAttention, cache ? cache[i] : undefined);
                x = output;
                if (includeAttention && attention) {
                    perLayerAtt.push(attention); // (B,T,T) already head-averaged
                }

                if (cache && newCache) {
                    cache[i]?.k.dispose();
                    cache[i]?.v.dispose();
                    cache[i] = newCache; // Update cache for this block
                } else if (newCache) {
                    newCache.k.dispose();
                    newCache.v.dispose();
                }
            }

            let aggregatedAttention: TF.Tensor | undefined;
            if (includeAttention && perLayerAtt.length > 0) {
                aggregatedAttention = this.computeAttentionRollout(perLayerAtt);
            }

            // Final layer norm
            x = this.lnF.apply(x) as TF.Tensor;

            // Embedding to logits
            const logits = this.wte.project(x) as TF.Tensor;

            let loss: TF.Tensor | undefined;
            if (targets) {
                loss = this.calculateLoss(logits, targets);
            }

            return { logits, loss, attention: includeAttention ? aggregatedAttention : undefined };
        });
    }

    generate(
        idx: TF.Tensor,
        cache?: (KVCache | undefined)[],
        options?: GenerateOptions
    ): { output: TF.Tensor; attention?: TF.Tensor; probabilities?: TF.Tensor } {
        const temperature = options?.temperature ?? 1.0;
        const topK = options?.topK;
        const usePadding = options?.usePadding ?? false;
        const includeAttention = options?.includeAttention ?? false;

        return this.tf.tidy(() => {
            const currentIdx = idx;

            // Crop sequence if it exceeds block size
            const seqLen = currentIdx.shape[1]!;
            const cropIdx =
                seqLen <= this.config.blockSize
                    ? currentIdx
                    : currentIdx.slice(
                          [0, seqLen - this.config.blockSize],
                          [currentIdx.shape[0], this.config.blockSize]
                      );
            const padding = usePadding ? this.config.blockSize - cropIdx.shape[1]! : 0;
            // In some cases padding is faster
            const padIdx =
                padding > 0
                    ? this.tf.pad(cropIdx, [
                          [0, 0],
                          [0, padding],
                      ])
                    : cropIdx;

            const { logits, attention } = this.forward(padIdx, undefined, false, includeAttention, cache);

            // Focus only on the last time step
            const lastTimeStep = logits.shape[1]! - 1 - padding;
            const lastLogits = logits.slice([0, lastTimeStep, 0], [logits.shape[0], 1, logits.shape[2]!]); // (b, 1, vocab_size)
            const lastAttention = attention
                ? attention.slice([0, lastTimeStep, 0], [attention.shape[0], 1, attention.shape[2]!])
                : undefined;

            const scaledLogits = lastLogits.div(temperature);

            let nextToken: TF.Tensor;

            if (topK) {
                const { values: topKValues, indices: topKIndices } = this.tf.topk(scaledLogits, topK);
                const sampledIdx = this.tf.multinomial(topKValues.squeeze([1]) as TF.Tensor1D, 1);
                nextToken = this.tf.gather(topKIndices.squeeze([1]), sampledIdx, 1);
            } else {
                nextToken = this.tf.multinomial(scaledLogits.squeeze([1]) as TF.Tensor1D, 1);
            }

            let probabilities: TF.Tensor | undefined;
            if (options?.includeProbabilities) {
                probabilities = this.tf.softmax(scaledLogits.squeeze([1]) as TF.Tensor1D);
            }

            nextToken = nextToken.reshape([1, 1]);
            return { output: nextToken, attention: lastAttention?.squeeze([1]), probabilities };
        });
    }

    getNumParams(): number {
        return estimateParameterCount(this.config);
    }

    dispose() {
        this.wte.dispose();
        if (this.wpe) this.wpe.dispose();
        this.drop.dispose();
        this.blocks.forEach((block) => block.dispose());
        this.lnF.dispose();
    }
}
