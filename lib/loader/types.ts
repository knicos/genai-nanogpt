import { LoRAConfig } from '@base/models/config';
import { TrainingState } from '@base/training/types';

export interface TransformersConfig {
    model_type: string;
    vocab_size: number;
    hidden_size: number;
    num_hidden_layers: number;
    num_attention_heads: number;
    block_size: number;
    dropout: number;
    biasInLinear: boolean;
    biasInLayerNorm: boolean;
    mlpFactor: number;
    useRope: boolean;
    loraConfig?: LoRAConfig;
}

export interface TransformersTokeniser {
    type: 'char' | 'bpe';
    vocab: string[];
    merges: [string, string][];
}

export interface TransformersMetadata {
    name?: string;
    version: number;
    application: string;
    training?: TrainingState;
    reference?: string; // Reference model
    url?: string; // Original URL if loaded from there
    [key: string]: unknown;
}
