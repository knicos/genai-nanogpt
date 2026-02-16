import { LoRAConfig } from '@base/models/config';
import { TrainingState } from '@base/training/types';

export interface TransformersConfigBase {
    model_type: 'GenAI_NanoGPT_v1' | 'GenAI_NanoGPT_v2';
    vocab_size: number;
    hidden_size: number;
    num_hidden_layers: number;
    num_attention_heads: number;
    block_size: number;
    mlpFactor: number;
    loraConfig?: LoRAConfig;
}

export interface TransformersConfigV1 extends TransformersConfigBase {
    model_type: 'GenAI_NanoGPT_v1';
    useRope: boolean;
}

export interface TransformersConfigV2 extends TransformersConfigBase {
    model_type: 'GenAI_NanoGPT_v2';
    windowSize?: string; // S or L for each layer, e.g. 'SSLLS' for 5 layers
}

export type TransformersConfig = TransformersConfigV1 | TransformersConfigV2;

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
