//import { PositionEmbedding } from './NLP/position_embedding';
import { defaultConfig, GPTConfig } from './config';
import Block from './layers/TransformerBlock';
import TiedEmbeddingOutputLayer from './layers/TiedEmbedding';
import { KVCache } from './layers/CausalSelfAttention';
import RoPECache from './layers/RoPECache';
import RMSNorm from './layers/RMSNorm';
import { estimateParameterCount } from './utilities/parameters';
import { createSoftmaxCrossEntropyWithGrad } from './training/sparseCrossEntropy';
import BaseLayer, { ForwardAttributes } from './layers/BaseLayer';
import { layers, initializers } from '@tensorflow/tfjs-layers';
import {
    add,
    eye,
    gather,
    keep,
    mod,
    multinomial,
    pad,
    range,
    scalar,
    softmax,
    Tensor,
    Tensor1D,
    tidy,
    topk,
} from '@tensorflow/tfjs-core';

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

export interface ModelForwardAttributes extends ForwardAttributes {
    includeAttention?: boolean;
    cache?: KVCache[];
    attentionOut?: Tensor;
    seed?: number;
}

// Main GPT model
export default class NanoGPT extends BaseLayer<ModelForwardAttributes> {
    private wte: TiedEmbeddingOutputLayer; // Token embeddings
    private wpe?: layers.Layer; // Position embeddings
    private drop: layers.Layer; // Dropout
    private blocks: Block[];
    private lnF: RMSNorm; // Final layer norm
    private ropeCache?: RoPECache;
    public log: TrainingLogEntry[] = []; // Training log

    constructor(config: Partial<GPTConfig> = {}) {
        super({ gpt: { ...defaultConfig, ...config }, layerConfig: {} });

        // Token embeddings
        this.wte = new TiedEmbeddingOutputLayer(this.config, 'token_embedding', this);

        if (this.config.gpt.useRope === false) {
            this.wpe = layers.embedding({
                inputDim: this.config.gpt.blockSize,
                outputDim: this.config.gpt.nEmbed,
                name: 'positional_embedding',
                embeddingsInitializer: initializers.randomNormal({ mean: 0.0, stddev: 0.02 }),
            });
        } else {
            this.ropeCache = new RoPECache(this.config.gpt);
            this.config.layerConfig.ropeCache = this.ropeCache;
        }

        this.drop = layers.dropout({ rate: this.config.gpt.dropout });

        // Transformer blocks
        this.blocks = [];
        for (let i = 0; i < this.config.gpt.nLayer; i++) {
            this.blocks.push(new Block(i, this.config, this));
        }

        // Final layer norm
        this.lnF = new RMSNorm(this.config, `final_rms_norm`, this);
    }

    get checkpointing(): boolean {
        return this.config.layerConfig.checkpointing === true;
    }

    set checkpointing(value: boolean) {
        this.config.layerConfig.checkpointing = value;
    }

    private inputPhase(idx: Tensor, pastLen: number, training: boolean = false): Tensor {
        return tidy(() => {
            //const [, seqLen] = idx.shape;
            //const maxCtx = this.config.blockSize;
            //const posStart = Math.min(pastLen, maxCtx - seqLen);

            const tokEmb = this.wte.embed(idx) as Tensor; // (b, t, n_embd)

            if (this.config.gpt.useRope === false) {
                const [, seqLen] = idx.shape;
                const maxCtx = this.config.gpt.blockSize;
                // position_ids = (pastLen + arange(T)) % maxCtx    // stays in [0, blockSize)
                const rng = range(0, seqLen, 1, 'int32'); // (t,)
                const posIdx = mod(add(rng, scalar(pastLen, 'int32')), scalar(maxCtx, 'int32')) as Tensor;
                const posEmb = this.wpe!.apply(posIdx) as Tensor; // (b, t, n_embd)

                const embSum = tokEmb.add(posEmb);

                const out = this.drop.apply(embSum, { training }) as Tensor;
                return out;
            } else {
                const out = this.drop.apply(tokEmb, { training }) as Tensor;
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

    private validateInput(idx: Tensor): void {
        if (idx.shape.length !== 2) {
            throw new Error(`Invalid input shape: expected [batch_size, sequence_length], got ${idx.shape}`);
        }
        if (idx.shape[1] > this.config.gpt.blockSize) {
            throw new Error(`Input sequence length ${idx.shape[1]} isn't block size ${this.config.gpt.blockSize}`);
        }
        if (idx.dtype !== 'int32') {
            throw new Error(`Input tensor must be of type int32, got ${idx.dtype}`);
        }
    }

    private calculateLoss(logits: Tensor, targets: Tensor): Tensor {
        try {
            //return this.tf.losses.softmaxCrossEntropy(targets, logits, this.tf.Reduction.MEAN);
            const lossFn = createSoftmaxCrossEntropyWithGrad();
            return lossFn(logits, targets).mean();
        } catch (error) {
            console.error('Error computing loss:', error);
            throw new Error(`Loss computation failed: ${error}`);
        }
    }

    // Attention rollout per Abnar & Zuidema (2020)
    // Expects list of (B, T, T) attention matrices already averaged over heads.
    private computeAttentionRollout(attentions: Tensor[]): Tensor {
        return tidy(() => {
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

            // Always slice to [B, Q, Q] for rollout
            const attentionsSliced = attentions.map((att) => att.slice([0, 0, 0], [B, Q, Q]));

            const ey = eye(Q, Q).expandDims(0); // (1,Q,Q)
            let rollout = ey.tile([B, 1, 1]); // (B,Q,Q)
            for (const att of attentionsSliced) {
                const a = att.add(ey);
                const aNorm = a.div(a.sum(-1, true)); // (B,Q,Q)
                rollout = aNorm.matMul(rollout); // (B,Q,Q)
            }
            return rollout;
        });
    }

    forward(attrs: ModelForwardAttributes, idx: Tensor, targets?: Tensor): Tensor[] {
        this.validateInput(idx);

        return tidy(() => {
            this.startMemory();
            // Token and position embeddings
            const pastLen = attrs.cache?.[0]?.length ?? 0;
            let x = this.inputPhase(idx, pastLen, attrs.training);

            const perLayerAtt: Tensor[] = [];

            if (attrs.cache && attrs.cache.length !== this.blocks.length) {
                console.error('Cache', attrs.cache);
                throw new Error(
                    `Cache length ${attrs.cache.length} does not match number of blocks ${this.blocks.length}`
                );
            }

            // Transformer blocks
            for (let i = 0; i < this.blocks.length; i++) {
                const block = this.blocks[i];
                const seed = Math.random() * 1e9;
                const blockAttrs = {
                    training: attrs.training,
                    seed,
                    includeAttention: attrs.includeAttention,
                    pastKV: attrs.cache ? attrs.cache[i] : undefined,
                    attentionOut: undefined as Tensor | undefined,
                };

                const output =
                    this.config.layerConfig.checkpointing && attrs.training
                        ? block.callCheckpoint(blockAttrs, x)
                        : block.call(blockAttrs, x);
                x.dispose();
                x = output as Tensor;

                if (attrs.includeAttention && blockAttrs.attentionOut) {
                    perLayerAtt.push(blockAttrs.attentionOut); // (B,T,T) already head-averaged
                }
            }

            let aggregatedAttention: Tensor | undefined;
            if (attrs.includeAttention && perLayerAtt.length > 0) {
                aggregatedAttention = this.computeAttentionRollout(perLayerAtt);
            }

            // Final layer norm
            x = this.lnF.call(attrs, x) as Tensor;

            // Embedding to logits
            const logits = this.wte.project(x) as Tensor;
            x.dispose();

            let loss: Tensor | undefined;
            if (targets) {
                loss = this.calculateLoss(logits, targets);
            }

            this.endMemory('Forward');

            attrs.attentionOut = aggregatedAttention ? keep(aggregatedAttention) : undefined;

            return loss ? [logits, loss] : [logits];
        });
    }

    generate(
        idx: Tensor,
        cache?: KVCache[],
        options?: GenerateOptions
    ): { output: Tensor; attention?: Tensor; probabilities?: Tensor } {
        const temperature = options?.temperature ?? 1.0;
        const tK = options?.topK;
        const usePadding = options?.usePadding ?? false;
        const includeAttention = options?.includeAttention ?? false;

        return tidy(() => {
            const currentIdx = idx;

            // Crop sequence if it exceeds block size
            const seqLen = currentIdx.shape[1]!;
            const cropIdx =
                seqLen <= this.config.gpt.blockSize
                    ? currentIdx
                    : currentIdx.slice(
                          [0, seqLen - this.config.gpt.blockSize],
                          [currentIdx.shape[0], this.config.gpt.blockSize]
                      );
            const padding = usePadding ? this.config.gpt.blockSize - cropIdx.shape[1]! : 0;
            // In some cases padding is faster
            const padIdx =
                padding > 0
                    ? pad(cropIdx, [
                          [0, 0],
                          [0, padding],
                      ])
                    : cropIdx;

            const attrs: ModelForwardAttributes = {
                training: false,
                includeAttention,
                cache,
            };
            const [logits] = this.forward(attrs, padIdx);

            // Focus only on the last time step
            const lastTimeStep = logits.shape[1]! - 1 - padding;
            const lastLogits = logits.slice([0, lastTimeStep, 0], [logits.shape[0], 1, logits.shape[2]!]); // (b, 1, vocab_size)
            const lastAttention = attrs.attentionOut
                ? attrs.attentionOut.slice(
                      [0, lastTimeStep, 0],
                      [attrs.attentionOut.shape[0], 1, attrs.attentionOut.shape[2]!]
                  )
                : undefined;
            logits.dispose();

            const scaledLogits = lastLogits.div(temperature);

            let nextToken: Tensor;

            if (tK) {
                const { values: topKValues, indices: topKIndices } = topk(scaledLogits, tK);
                const sampledIdx = multinomial(topKValues.squeeze([1]) as Tensor1D, 1);
                nextToken = gather(topKIndices.squeeze([1]), sampledIdx, 1);
            } else {
                nextToken = multinomial(scaledLogits.squeeze([1]) as Tensor1D, 1);
            }

            let probabilities: Tensor | undefined;
            if (options?.includeProbabilities) {
                probabilities = softmax(scaledLogits.squeeze([1]) as Tensor1D);
            }

            nextToken = nextToken.reshape([1, 1]);
            return { output: nextToken, attention: lastAttention?.squeeze([1]), probabilities };
        });
    }

    getNumParams(): number {
        return estimateParameterCount(this.config.gpt);
    }

    dispose() {
        this.wte.dispose();
        if (this.wpe) this.wpe.dispose();
        this.drop.dispose();
        this.blocks.forEach((block) => block.dispose());
        this.lnF.dispose();
    }
}
