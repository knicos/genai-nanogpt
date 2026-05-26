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
    public metaData: TransformersMetadata = { version: 2, application: '@genai-fi/nanogpt' };
    private loraLayer?: LoRA;
    private loraMap = new Map<string, LoRA>();

    constructor(config: C) {
        super(config);
        if (config.loraConfig) {
            config.loraConfig.forEach((loraConfig, name) => {
                this.createLoRA(name, loraConfig);
            });
        }
    }

    createLoRA(name: string, loraConfig: LoRAConfig) {
        if (this.loraMap.has(name)) {
            //throw new Error(`LoRA with name ${name} already exists.`);
            return;
        }
        const lora = new LoRA(name, this.weightStore, loraConfig.alpha, loraConfig.rank, loraConfig.variables);
        this.loraMap.set(name, lora);

        this.config.loraConfig = this.config.loraConfig || new Map<string, LoRAConfig>();
        this.config.loraConfig.set(name, loraConfig);
    }

    deleteLoRA(name: string) {
        const lora = this.loraMap.get(name);
        if (!lora) {
            throw new Error(`No LoRA with name ${name} exists.`);
        }
        if (this.loraLayer === lora) {
            this.detachLoRA();
        }
        lora.dispose();
        this.loraMap.delete(name);

        if (this.config.loraConfig) {
            this.config.loraConfig.delete(name);
        }
    }

    renameLoRA(oldName: string, newName: string) {
        if (!this.loraMap.has(oldName)) {
            throw new Error(`No LoRA with name ${oldName} exists.`);
        }
        if (this.loraMap.has(newName)) {
            throw new Error(`LoRA with name ${newName} already exists.`);
        }
        const lora = this.loraMap.get(oldName)!;
        this.loraMap.set(newName, lora);
        this.loraMap.delete(oldName);
        if (this.config.loraConfig) {
            this.config.loraConfig.delete(oldName);
            this.config.loraConfig.set(newName, {
                rank: lora.rank,
                alpha: lora.alpha,
                variables: Array.from(lora.variables),
            });
        }
    }

    mergeLoRA(name: string) {
        const lora = this.loraMap.get(name);
        if (!lora) {
            throw new Error(`No LoRA with name ${name} exists.`);
        }
        lora.merge();
        this.deleteLoRA(name);
    }

    attachLoRA(name: string) {
        if (this.loraLayer) {
            if (this.loraLayer.name === name) {
                return; // Already attached
            }
            this.detachLoRA();
        }
        const lora = this.loraMap.get(name);
        if (!lora) {
            throw new Error(`No LoRA with name ${name} exists.`);
        }
        lora.attach();
        this.loraLayer = lora;
        this.config.loraName = name;
    }

    detachLoRA() {
        if (!this.loraLayer) {
            throw new Error('No LoRA layer is attached to this model.');
        }
        this.loraLayer.detach();
        this.loraLayer = undefined;
        this.config.loraName = undefined;
    }

    hasLoRA(name?: string): boolean {
        if (name) {
            return this.loraMap.has(name);
        }
        return !!this.loraLayer;
    }

    listLoRAs(): string[] {
        return Array.from(this.loraMap.keys());
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
