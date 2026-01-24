import '@tensorflow/tfjs';

export { default as NanoGPT } from './models/NanoGPTV1';
export { default as TeachableLLM } from './TeachableLLM';
export { default as CharTokeniser } from './tokeniser/CharTokeniser';
export { default as BPETokeniser } from './tokeniser/bpe';
export { default as waitForModel } from './utilities/waitForModel';
export { default as loadTextData } from './data/textLoader';
export { default as Generator } from './Generator';
export type { ITrainerOptions } from './Trainer';
export type { IGenerateOptions } from './Generator';
export { type ModelForwardAttributes, default as Model } from './models/model';
export type { ITokeniser, Conversation, Roles } from './tokeniser/type';
export type { TrainingProgress, TrainingLogEntry } from './training/Trainer';
export type { GPTConfig } from './models/config';
export {
    estimateParameterCount,
    estimateMemoryUsage,
    estimateTrainingMemoryUsage,
    estimateResources,
    validateConfig,
} from './utilities/parameters';
export { default as topP } from './utilities/topP';
export { Task, tokensFromTasks } from './training/tasks/Task';
import { default as PretrainingTask } from './training/tasks/PretrainingTask';
import { default as StartSentenceTask } from './training/tasks/StartSentenceTask';
import { default as ConversationTask } from './training/tasks/ConversationTask';

export const tasks = {
    PretrainingTask,
    StartSentenceTask,
    ConversationTask,
};

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

const ops = {
    pack16,
    unpack16,
};

export { ops };

export { selectBackend } from './backend';
export { default as performanceTest } from './utilities/performance';

import CausalSelfAttention from './layers/CausalSelfAttention';
import MLP from './layers/MLP';
import TransformerBlock from './layers/TransformerBlock';
import RoPECache from './layers/RoPECache';

export const layers = {
    CausalSelfAttention,
    MLP,
    TransformerBlock,
    RoPECache,
};

export { default as AdamExt } from './training/AdamExt';

export { default as checks } from './checks';
export type { TensorStatistics } from './checks/weights';
export { sentenceEmbeddings, sentenceEmbeddingsTensor } from './utilities/sentences';
