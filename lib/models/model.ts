import { Tensor } from '@tensorflow/tfjs-core';
import type { ForwardAttributes } from '../layers/BaseLayer';
import type { AttentionScores, KVCache } from '../layers/CausalSelfAttention';
import BaseLayer from '../layers/BaseLayer';
import { estimateParameterCount } from '../main';

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
export default abstract class Model<T extends ModelForwardAttributes> extends BaseLayer<T> {
    public lossScaling = 128;
    public trainingState: TrainingState | null = null;

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
