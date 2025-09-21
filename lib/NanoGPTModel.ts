//import { PositionEmbedding } from './NLP/position_embedding';
import { defaultConfig, GPTConfig } from './config';
import Block from './layers/TransformerBlock';
import TiedEmbeddingOutputLayer from './layers/TiedEmbedding';
import { KVCache } from './layers/CausalSelfAttention';
import RoPECache from './layers/RoPECache';
import RMSNorm from './layers/RMSNorm';
import { estimateParameterCount } from './utilities/parameters';
import { createSoftmaxCrossEntropyWithGrad } from './training/sparseCrossEntropy';
import BaseLayer from './layers/BaseLayer';
import { layers, initializers } from '@tensorflow/tfjs-layers';
import {
    add,
    eye,
    gather,
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
    Variable,
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

// Main GPT model
export default class NanoGPT extends BaseLayer {
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
        this.wte = new TiedEmbeddingOutputLayer({
            vocabSize: this.config.gpt.vocabSize,
            embedDim: this.config.gpt.nEmbed,
            name: 'token_embedding',
        });

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
            this.blocks.push(new Block(i, this.config));
        }

        // Final layer norm
        this.lnF = new RMSNorm(this.config, 1e-8, `final_rms_norm`);
    }

    get checkpointing(): boolean {
        return this.config.layerConfig.checkpointAttention === true || this.config.layerConfig.checkpointMLP === true;
    }

    set checkpointing(value: boolean) {
        this.config.layerConfig.checkpointAttention = value;
        this.config.layerConfig.checkpointMLP = value;
    }

    get variables(): Variable[] {
        return [
            //...this.wpe.trainableWeights.map((v) => v.read() as TF.Variable),
            ...this.blocks.flatMap((b) => b.variables),
            ...this.lnF.trainableWeights.map((v) => v as Variable),
            ...this.wte.variables,
        ];
    }

    public saveWeights(): Map<string, Tensor[]> {
        const map = new Map<string, Tensor[]>();
        map.set('token_embedding', this.wte.getWeights());
        if (this.wpe) map.set('positional_embedding', this.wpe.getWeights());
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].saveWeights(map);
        }
        map.set('final_rms_norm', this.lnF.getWeights());
        return map;
    }

    public loadWeights(weights: Map<string, Tensor[]>): void {
        this.wte.setWeights(weights.get('token_embedding') || []);
        if (this.wpe) this.wpe.setWeights(weights.get('positional_embedding') || []);
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].loadWeights(weights);
        }
        this.lnF.setWeights(weights.get('final_rms_norm') || []);
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

    set trainable(value: boolean) {
        for (const block of this.blocks) {
            block.trainable = value;
        }

        //this.wpe.trainable = value;
        this.lnF.trainable = value;
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

            if (Q === K) {
                // Full square attentions: standard rollout
                const ey = eye(K, K).expandDims(0); // (1,K,K)
                let rollout = ey.tile([B, 1, 1]); // (B,K,K)
                for (const att of attentions) {
                    const a = att.add(ey);
                    const aNorm = a.div(a.sum(-1, true)); // (B,K,K)
                    rollout = aNorm.matMul(rollout); // (B,K,K)
                }
                return rollout;
            }

            throw new Error(`Unsupported attention shapes for rollout: [B=${B}, Q=${Q}, K=${K}]`);
        });
    }

    forward(
        idx: Tensor,
        targets?: Tensor,
        training = false,
        includeAttention = false,
        cache?: (KVCache | undefined)[]
    ): { logits: Tensor; loss?: Tensor; attention?: Tensor } {
        this.validateInput(idx);

        return tidy(() => {
            this.startMemory();
            // Token and position embeddings
            const pastLen = cache?.[0]?.length ?? 0;
            let x = this.inputPhase(idx, pastLen, training);

            const perLayerAtt: Tensor[] = [];

            if (cache && cache.length !== this.blocks.length) {
                console.error('Cache', cache);
                throw new Error(`Cache length ${cache.length} does not match number of blocks ${this.blocks.length}`);
            }

            // Transformer blocks
            for (let i = 0; i < this.blocks.length; i++) {
                const oldX = x;

                const block = this.blocks[i];
                const {
                    output,
                    attention,
                    cache: newCache,
                } = block.call(x, training, includeAttention, cache ? cache[i] : undefined);
                x = output;
                oldX.dispose();
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

            let aggregatedAttention: Tensor | undefined;
            if (includeAttention && perLayerAtt.length > 0) {
                aggregatedAttention = this.computeAttentionRollout(perLayerAtt);
            }

            // Final layer norm
            x = this.lnF.apply(x) as Tensor;

            // Embedding to logits
            const logits = this.wte.project(x) as Tensor;

            let loss: Tensor | undefined;
            if (targets) {
                loss = this.calculateLoss(logits, targets);
            }

            this.endMemory('Forward');

            return { logits, loss, attention: includeAttention ? aggregatedAttention : undefined };
        });
    }

    generate(
        idx: Tensor,
        cache?: (KVCache | undefined)[],
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

            const { logits, attention } = this.forward(padIdx, undefined, false, includeAttention, cache);

            // Focus only on the last time step
            const lastTimeStep = logits.shape[1]! - 1 - padding;
            const lastLogits = logits.slice([0, lastTimeStep, 0], [logits.shape[0], 1, logits.shape[2]!]); // (b, 1, vocab_size)
            const lastAttention = attention
                ? attention.slice([0, lastTimeStep, 0], [attention.shape[0], 1, attention.shape[2]!])
                : undefined;

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
