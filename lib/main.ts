export { default as NanoGPT } from './NanoGPTModel';
export { default as TeachableLLM } from './TeachableLLM';
export { default as CharTokeniser } from './tokeniser/CharTokeniser';
export { default as BPETokeniser } from './tokeniser/bpe';
export { default as waitForModel } from './utilities/waitForModel';
export { default as loadTextData } from './data/textLoader';
export type { ITrainerOptions } from './Trainer';
export type { IGenerateOptions } from './Generator';
export type { TrainingLogEntry } from './NanoGPTModel';
export type { ITokeniser } from './tokeniser/type';
export type { TrainingProgress } from './training/Trainer';
export type { GPTConfig } from './config';
export {
    estimateParameterCount,
    estimateMemoryUsage,
    estimateTrainingMemoryUsage,
    estimateResources,
    validateConfig,
} from './utilities/parameters';

import './ops/scatterSub';
import './ops/gatherSub';
import './ops/attentionMask';
import './ops/qkv';
import './ops/rope';
import './ops/appendCache';
import './ops/fusedSoftmax';
import './ops/matMulGelu';
import './ops/gelu';
import './ops/normRMS';
import './ops/log';

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
