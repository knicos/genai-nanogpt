import { defaultConfig, GPTConfig } from './config';
import Block from '../layers/TransformerBlock';
import TiedEmbeddingOutputLayer from '../layers/TiedEmbedding';
import RoPECache from '../layers/RoPECache';
import RMSNorm from '../layers/RMSNorm';
import { keep, Tensor, tidy } from '@tensorflow/tfjs-core';
import Model, { ModelForwardAttributes } from './model';
import PositionEmbedding from '@base/layers/PositionEmbedding';

// Main NanoGPT model
export default class NanoGPT extends Model<ModelForwardAttributes> {
    private wte: TiedEmbeddingOutputLayer; // Token embeddings
    private wpe?: PositionEmbedding; // Position embeddings
    // private drop: layers.Layer; // Dropout
    private blocks: Block[];
    private lnF: RMSNorm; // Final layer norm
    private ropeCache?: RoPECache;

    constructor(config: Partial<GPTConfig> = {}) {
        super({ ...defaultConfig, ...config });

        // Token embeddings
        this.wte = new TiedEmbeddingOutputLayer(this.config, 'token_embedding', this);

        if (this.config.useRope === false) {
            // Absolute positional embeddings
            this.wpe = new PositionEmbedding(this.config, 'positional_embedding', this);
        } else {
            this.ropeCache = new RoPECache(this.config);
        }

        // this.drop = layers.dropout({ rate: this.config.dropout });

        // Transformer blocks
        this.blocks = [];
        for (let i = 0; i < this.config.nLayer; i++) {
            this.blocks.push(new Block(i, this.config, this));
        }

        // Final layer norm
        this.lnF = new RMSNorm(this.config, `final_rms_norm`, this);
    }

    getClassName() {
        return 'GenAI_NanoGPT_v1';
    }

    private inputPhase(idx: Tensor, attrs: ModelForwardAttributes): Tensor {
        return tidy(() => {
            const tokEmb = this.wte.embed(idx) as Tensor; // (b, t, n_embd)

            if (this.config.useRope === false) {
                const out = this.wpe!.call(attrs, tokEmb);
                if (Array.isArray(out)) {
                    throw new Error('PositionEmbedding output should not be an array');
                }
                return out;
            } /* else {
                const out = this.drop.apply(tokEmb, { training }) as Tensor;
                return out;
            }*/
            return tokEmb;
        });
    }

    forward(attrs: ModelForwardAttributes, idx: Tensor, targets?: Tensor): Tensor[] {
        this.validateInput(idx);

        attrs.ropeCache = this.ropeCache;

        if (attrs.outputEmbeddings) {
            attrs.embeddings = [];
        }

        return tidy(() => {
            this.startMemory();
            // Token and position embeddings
            let x = this.inputPhase(idx, attrs);

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
                    ...attrs,
                    seed,
                    pastKV: attrs.cache ? attrs.cache[i] : undefined,
                };

                const output =
                    attrs.checkpointing && attrs.training
                        ? block.callCheckpoint(blockAttrs, x)
                        : block.call(blockAttrs, x);

                if (attrs.outputEmbeddings) {
                    keep(x);
                    attrs.embeddings!.push({ name: `block_output_${i}`, tensor: x });
                } else {
                    x.dispose();
                }
                x = output as Tensor;
            }

            // Final layer norm
            x = this.lnF.call(attrs, x) as Tensor;

            // Embedding to logits
            const logits = this.wte.project(x) as Tensor;
            if (attrs.outputEmbeddings) {
                keep(x);
                attrs.embeddings!.push({ name: `final_norm_output`, tensor: x });
            } else {
                x.dispose();
            }

            let loss: Tensor | undefined;
            if (targets) {
                loss = this.calculateLoss(logits, targets);
            }

            this.endMemory('Forward');

            return loss ? [logits, loss] : [logits];
        });
    }

    project(embeddings: Tensor): Tensor {
        return tidy(() => {
            const logits = this.wte.project(embeddings) as Tensor;
            return logits;
        });
    }

    dispose() {
        this.wte.dispose();
        if (this.wpe) this.wpe.dispose();
        // this.drop.dispose();
        this.blocks.forEach((block) => block.dispose());
        this.lnF.dispose();
    }
}
