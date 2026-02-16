import { Tensor } from '@tensorflow/tfjs-core';
import type { ForwardAttributes } from '../layers/BaseLayer';
import type { AttentionScores, KVCache } from '../layers/CausalSelfAttention';
import BaseLayer from '../layers/BaseLayer';
import { estimateParameterCount } from '../main';
import { TransformersMetadata } from '@base/loader/types';
import { GPTConfig, LoRAConfig } from './config';
import LoRA from '@base/layers/LoRA';

export interface ModelForwardAttributes extends ForwardAttributes {
    cache?: KVCache[];
    attentionScores?: AttentionScores;
    seed?: number;
    skipLogits?: boolean; // Whether to output embeddings instead of logits
}

interface TrainingState {
    steps: number;
    learningRate: number;
    batchSize: number;
    loss: number;
}

// Abstract base class for models
export default abstract class Model<
    T extends ModelForwardAttributes,
    C extends GPTConfig = GPTConfig,
> extends BaseLayer<T, C> {
    public lossScaling = 128;
    public trainingState: TrainingState | null = null;
    public metaData?: TransformersMetadata;
    private loraLayer?: LoRA;

    /*constructor(config: GPTConfig) {
        super(config);
        if (config.loraConfig) {
            console.log('Attaching LoRA layer with config:', config.loraConfig);
            this.attachLoRA(config.loraConfig);
        }
    }*/

    attachLoRA(loraConfig: LoRAConfig) {
        if (this.loraLayer) {
            throw new Error('LoRA is already attached to this model.');
        }
        this.config.loraConfig = loraConfig;
        this.loraLayer = new LoRA(this.weightStore, loraConfig.alpha, loraConfig.rank, loraConfig.variables);
    }

    detachLoRA() {
        if (!this.loraLayer) {
            throw new Error('No LoRA layer is attached to this model.');
        }
        this.loraLayer.dispose();
        this.loraLayer = undefined;
        delete this.config.loraConfig;
    }

    hasLoRA(): boolean {
        return !!this.loraLayer;
    }

    get lora(): LoRA | null {
        return this.loraLayer || null;
    }

    abstract getClassName(): string;

    abstract forward(attrs: T, idx: Tensor): Tensor;

    abstract project(embeddings: Tensor): Tensor;

    abstract dispose(): void;

    getNumParams(): number {
        return estimateParameterCount(this.config);
    }

    protected validateInput(idx: Tensor): void {
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
}
