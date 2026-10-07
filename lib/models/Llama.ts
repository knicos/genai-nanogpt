import { GPTConfigLlama } from './config';
import Block, { TransformerBlockConfig } from '../layers/TransformerBlock';
import TiedEmbeddingOutputLayer from '../layers/TiedEmbedding';
import RoPECache from '../layers/RoPECache';
import RMSNorm from '../layers/RMSNorm';
import { keep, Tensor, tidy } from '@tensorflow/tfjs-core';
import Model, { ModelForwardAttributes } from './model';
import { packingSupported } from '@base/utilities/packed';
import { pack16 } from '@base/ops/pack16';
import { unpack16 } from '@base/ops/unpack16';
import HeadLayer from '@base/layers/HeadLayer';

const defaultConfig: GPTConfigLlama = {
    modelType: 'llama',
    vocabSize: 2000,
    blockSize: 128, // Maximum sequence length
    nLayer: 6, // Number of transformer layers
    nHead: 4, // Number of attention heads
    nEmbed: 256, // Embedding dimension
    mlpFactor: 4,
};

// Main NanoGPT model
export default class Llama extends Model<ModelForwardAttributes, GPTConfigLlama> {
    private wte: TiedEmbeddingOutputLayer; // Token embeddings
    private head: HeadLayer;

    private blocks: Block[];
    private lnF: RMSNorm; // Final layer norm
    private ropeCache?: RoPECache;

    constructor(config: Partial<GPTConfigLlama> = {}) {
        super({ ...defaultConfig, ...config });

        const blockConfig: TransformerBlockConfig = {
            activation: 'swiglu',
            hiddenSize: Math.max(256, Math.ceil((this.config.nEmbed * (8 / 3)) / 256) * 256),
            useGamma: true,
            useQKNorm: false,
            ropeInterleaved: false,
        };

        // Token embeddings
        this.wte = new TiedEmbeddingOutputLayer(this.config, 'token_embedding', this);
        this.ropeCache = new RoPECache(this.config);

        // Transformer blocks
        this.blocks = [];
        for (let i = 0; i < this.config.nLayer; i++) {
            this.blocks.push(new Block(i, this.config, blockConfig, this));
        }

        // Final layer norm
        this.lnF = new RMSNorm(this.config, blockConfig, `final_rms_norm`, this);

        // Head layer
        this.head = new HeadLayer(this.config, 'lm_head', this);
    }

    getClassName() {
        return 'llama';
    }

    private inputPhase(idx: Tensor): Tensor {
        return tidy(() => {
            const tokEmb = this.wte.embed(idx) as Tensor; // (b, t, n_embd)
            return tokEmb;
        });
    }

    forward(attrs: ModelForwardAttributes, idx: Tensor): Tensor {
        this.validateInput(idx);

        attrs.ropeCache = this.ropeCache;

        if (attrs.outputEmbeddings) {
            attrs.embeddings = [];
        }

        return tidy(() => {
            this.startMemory();

            let x = this.inputPhase(idx);

            if (attrs.cache && attrs.cache.length !== this.blocks.length) {
                console.error('Cache', attrs.cache);
                throw new Error(
                    `Cache length ${attrs.cache.length} does not match number of blocks ${this.blocks.length}`
                );
            }

            const usedMixed = attrs.mixedPrecision === true && packingSupported();

            let pX = usedMixed ? pack16(x) : x;
            if (usedMixed && x !== pX) {
                x.dispose();
            }

            // Transformer blocks
            for (let i = 0; i < this.blocks.length; i++) {
                if (attrs.layerDrop && Math.random() < attrs.layerDrop * (i / this.blocks.length)) {
                    continue; // Skip this layer
                }
                const block = this.blocks[i];
                const seed = Math.random() * 1e9;
                const blockAttrs = {
                    ...attrs,
                    seed,
                    pastKV: attrs.cache ? attrs.cache[i] : undefined,
                    mixedPrecision: usedMixed,
                    epsilon: 1e-6, // For RMSNorm
                };

                const output =
                    attrs.checkpointing && attrs.training
                        ? block.callCheckpoint(blockAttrs, pX)
                        : block.call(blockAttrs, pX);

                if (attrs.outputEmbeddings) {
                    keep(pX);
                    attrs.embeddings!.push({ name: `block_output_${i}`, tensor: pX });
                } else {
                    pX.dispose();
                }
                pX = output as Tensor;
            }

            // Final layer norm
            x = this.lnF.call({ ...attrs, mixedPrecision: usedMixed }, pX) as Tensor;
            pX.dispose();

            if (attrs.skipLogits) {
                this.endMemory('Forward');
                return x;
            }

            // Embedding to logits
            const packedLogits = this.head.project(x) as Tensor;

            if (attrs.outputEmbeddings) {
                keep(x);
                attrs.embeddings!.push({ name: `final_norm_output`, tensor: x });
            } else {
                x.dispose();
            }

            const logits = usedMixed ? unpack16(packedLogits) : packedLogits;
            if (usedMixed && packedLogits !== logits) {
                packedLogits.dispose();
            }

            return logits;
        });
    }

    project(embeddings: Tensor): Tensor {
        return tidy(() => {
            const logits = this.head.project(embeddings) as Tensor;
            return logits;
        });
    }

    dispose() {
        this.weightStore.dispose();
        this.wte.dispose();
        this.head.dispose();

        this.blocks.forEach((block) => block.dispose());
        this.lnF.dispose();
    }
}
