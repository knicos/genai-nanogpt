import type TF from '@tensorflow/tfjs';
//import { PositionEmbedding } from './NLP/position_embedding';
import { defaultConfig, GPTConfig } from './config';
import Block from './layers/TransformerBlock';
import TiedEmbeddingOutputLayer from './layers/TiedEmbedding';
import LayerNorm from './layers/LayerNorm';

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
}

// Main GPT model
export default class NanoGPT {
    public readonly config: GPTConfig;
    private wte: TiedEmbeddingOutputLayer; // Token embeddings
    private wpe: TF.layers.Layer; // Position embeddings
    private drop: TF.layers.Layer; // Dropout
    private blocks: Block[];
    private lnF: LayerNorm; // Final layer norm
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

        this.wpe = this.tf.layers.embedding({
            inputDim: this.config.blockSize,
            outputDim: this.config.nEmbed,
            name: 'positional_embedding',
            embeddingsInitializer: this.tf.initializers.randomNormal({ mean: 0.0, stddev: 0.02 }),
        });

        this.drop = this.tf.layers.dropout({ rate: this.config.dropout });

        // Transformer blocks
        this.blocks = [];
        for (let i = 0; i < this.config.nLayer; i++) {
            this.blocks.push(new Block(this.tf, i, this.config));
        }

        // Final layer norm
        this.lnF = new LayerNorm(tf, [this.config.nEmbed], 1e-5, `final_layer_norm`);
    }

    get variables(): TF.Variable[] {
        return [
            ...this.wpe.trainableWeights.map((v) => v.read() as TF.Variable),
            ...this.blocks.flatMap((b) => b.variables),
            ...this.lnF.trainableWeights.map((v) => v as TF.Variable),
            ...this.wte.variables,
        ];
    }

    public saveWeights(): Map<string, TF.Tensor[]> {
        const map = new Map<string, TF.Tensor[]>();
        map.set('token_embedding', this.wte.getWeights());
        map.set('positional_embedding', this.wpe.getWeights());
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].saveWeights(map);
        }
        map.set('final_layer_norm', this.lnF.getWeights());
        return map;
    }

    public loadWeights(weights: Map<string, TF.Tensor[]>): void {
        this.wte.setWeights(weights.get('token_embedding') || []);
        this.wpe.setWeights(weights.get('positional_embedding') || []);
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].loadWeights(weights);
        }
        this.lnF.setWeights(weights.get('final_layer_norm') || []);
    }

    private inputPhase(idx: TF.Tensor, training: boolean = false): TF.Tensor {
        return this.tf.tidy(() => {
            const [, seqLen] = idx.shape;

            const tokEmb = this.wte.embed(idx) as TF.Tensor; // (b, t, n_embd)
            const posIndices = this.tf.range(0, seqLen, 1, 'int32');
            const posEmb = this.wpe.apply(posIndices) as TF.Tensor; // (b, t, n_embd)

            const embSum = tokEmb.add(posEmb);

            const out = this.drop.apply(embSum, { training }) as TF.Tensor;
            return out;
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

        this.wpe.trainable = value;
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
            const B = attentions[0].shape[0]!;
            const T = attentions[0].shape[1]!;
            const eye = this.tf.eye(T, T).expandDims(0); // (1,T,T)
            let rollout = eye.tile([B, 1, 1]); // (B,T,T) start as identity
            for (const att of attentions) {
                // Add residual path
                let a = att.add(eye); // broadcast add
                // Row-normalize
                a = a.div(a.sum(-1, true));
                // Propagate
                rollout = a.matMul(rollout);
            }
            return rollout;
        });
    }

    forward(
        idx: TF.Tensor,
        targets?: TF.Tensor,
        training = false,
        includeAttention = false
    ): { logits: TF.Tensor; loss?: TF.Tensor; attention?: TF.Tensor } {
        this.validateInput(idx);

        return this.tf.tidy(() => {
            // Token and position embeddings
            let x = this.inputPhase(idx, training);

            const perLayerAtt: TF.Tensor[] = [];

            // Transformer blocks
            for (const block of this.blocks) {
                const { output, attention } = block.call(x, training, includeAttention);
                x = output;
                if (includeAttention && attention) {
                    perLayerAtt.push(attention); // (B,T,T) already head-averaged
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

    generate(idx: TF.Tensor, options?: GenerateOptions): { output: TF.Tensor; attention?: TF.Tensor } {
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

            const { logits, attention } = this.forward(padIdx, undefined, false, includeAttention);

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
                /*const probs = this.tf.softmax(scaledLogits).squeeze();
                const probsArray = probs.arraySync() as number[];
                const tokenPairs = probsArray.map((prob, idx) => ({ token: idx, prob }));
                tokenPairs.sort((a, b) => b.prob - a.prob); // Sort probabilities for debugging*/

                nextToken = this.tf.multinomial(scaledLogits.squeeze([1]) as TF.Tensor1D, 1);
            }

            nextToken = nextToken.reshape([1, 1]);
            return { output: nextToken, attention: lastAttention?.squeeze([1]) };
        });
    }

    getNumParams(): number {
        const embeddingParams = this.config.vocabSize * this.config.nEmbed + this.config.blockSize * this.config.nEmbed;
        const attentionParams =
            this.config.nLayer *
            (4 * this.config.nEmbed * this.config.nEmbed + // qkv + proj
                2 * this.config.nEmbed); // layer norms
        const mlpParams =
            this.config.nLayer *
            (4 * this.config.nEmbed * this.config.nEmbed + // fc
                this.config.nEmbed * 4 * this.config.nEmbed); // proj
        const finalParams = this.config.nEmbed + this.config.vocabSize * this.config.nEmbed;

        return embeddingParams + attentionParams + mlpParams + finalParams;
    }
}
