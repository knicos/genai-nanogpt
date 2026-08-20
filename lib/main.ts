import '@tensorflow/tfjs';

// Main API
export { default as TeachableLLM } from './TeachableLLM';

// Tokenisers
export * as tokenise from './tokenise';
export type { ITokeniser, Conversation, Roles } from './tokeniser/type';

// Data
export * as data from './data';
export type { ConversationStream } from './data/stream';
export type { DatasetMetadata, ModelMode } from './loader/types';

// Models
export * as models from './models';
export type { GPTConfig } from './models/config';
export type { ModelForwardAttributes } from './models/model';
export type { IGenerateOptions, IGeneratorResponse, IGeneratorOutput, GeneratorConversation } from './inference/types';
export type { TrainingOptions, TrainingLogEntry } from './training/types';
export type { ITrainingJob } from './api/training';

// Training
import { default as Evaluator } from './training/Evaluator';
import { AdamWOptimizer } from './training/AdamW';
export const training = {
    Evaluator,
    AdamWOptimizer,
};

// Utilities
import {
    estimateParameterCount,
    estimateMemoryUsage,
    estimateTrainingMemoryUsage,
    estimateResources,
    validateConfig,
} from './utilities/parameters';
import { default as topP } from './utilities/topP';
import { sliceUint16Shards, sliceUint8Shards } from './utilities/tokens';
import { default as performanceTest } from './utilities/performance';
import { sentenceEmbeddings, sentenceEmbeddingsTensor } from './utilities/sentences';

export const utilities = {
    estimateParameterCount,
    estimateMemoryUsage,
    estimateTrainingMemoryUsage,
    estimateResources,
    validateConfig,
    topP,
    sliceUint16Shards,
    sliceUint8Shards,
    performanceTest,
    sentenceEmbeddings,
    sentenceEmbeddingsTensor,
};

// Ops
import './ops/scatterSub';
import './ops/gatherSub';
import './ops/attentionMask';
import './ops/qkv';
import './ops/rope';
import './ops/appendCache';
import './ops/matMulGelu';
import './ops/gelu';
import './ops/normRMS';
import './ops/log';
import './ops/adamMoments';
import './ops/adamAdjust';
import { pack16 } from './ops/pack16';
import { unpack16 } from './ops/unpack16';
import './ops/softmax16';
import './ops/matMul16';
import './ops/transpose16';

export const ops = {
    pack16,
    unpack16,
};

// Layers
export * as layers from './layers';

// Checks
export { default as checks } from './checks';
export type { TensorStatistics } from './checks/weights';
