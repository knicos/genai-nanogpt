import { GPTConfig, GPTConfigV1, GPTConfigV2, GPTConfigQwen3, GPTConfigLlama } from '@base/models/config';
import Llama from './Llama';
import NanoGPT from './NanoGPTV1';
import Model, { ModelForwardAttributes } from './model';
import NanoGPTV2 from './NanoGPTV2';
import Qwen3 from './Qwen3';

export default function createModelInstance(config: GPTConfig): Model<ModelForwardAttributes, GPTConfig> {
    const modelType = config.modelType || 'GenAI_NanoGPT_v1';
    switch (modelType) {
        case 'GenAI_NanoGPT_v1':
            return new NanoGPT(config as GPTConfigV1);
        case 'GenAI_NanoGPT_v2':
            return new NanoGPTV2(config as GPTConfigV2);
        case 'qwen3':
            return new Qwen3(config as GPTConfigQwen3);
        case 'llama':
            return new Llama(config as GPTConfigLlama);
        default:
            throw new Error(`Unsupported model type: ${modelType}`);
    }
}
