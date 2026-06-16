import { GenerateOptions } from '@base/inference/types';
import { LoRAConfig } from '@base/models/config';
import Model, { ModelForwardAttributes, TrainingState } from '@base/models/model';
import { ITokeniser } from '@base/tokeniser/type';
import { AdamWOptimizer } from '@base/training/AdamW';
import { TrainingLogEntry, TrainingOptions } from '@base/training/types';

export interface TransformersConfigBase {
    model_type: 'GenAI_NanoGPT_v1' | 'GenAI_NanoGPT_v2';
    vocab_size: number;
    hidden_size: number;
    num_hidden_layers: number;
    num_attention_heads: number;
    block_size: number;
    mlpFactor: number;
    loraConfig?: Record<string, LoRAConfig>;
    loraName?: string;
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
    datasetID?: string;
    id?: string;
}

export type ModelMode = 'untrained' | 'completion' | 'conversational';

export interface DatasetMetadata {
    id: string;
    name: string;
    conversational: boolean;
}

export interface ActionLogEntry {
    action: 'pretrain' | 'generate' | 'finetune';
    timestamp: number;
    duration: number;
    tokensProcessed: number;
    options: TrainingOptions | GenerateOptions;
}

export interface TransformersMetadata {
    name?: string;
    version: number;
    application: string;
    training?: TrainingState;
    reference?: string; // Reference model
    id?: string;
    url?: string; // Original URL if loaded from there
    mode?: ModelMode;
    pretrainingData?: DatasetMetadata[];
    pretrainingSettings?: TrainingOptions; // Last used training settings for pretraining
    generationSettings?: GenerateOptions;
    actionLog?: ActionLogEntry[];
    [key: string]: unknown;
}

export interface LoadResult {
    model: Model<ModelForwardAttributes>;
    tokeniser: ITokeniser;
    metaData: TransformersMetadata;
    optimizer?: AdamWOptimizer;
    log?: TrainingLogEntry[];
}
