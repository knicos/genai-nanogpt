import { GPTConfig, GPTConfigV1, GPTConfigV2 } from '@base/models/config';
import NanoGPT from './NanoGPTV1';
import Model, { ModelForwardAttributes } from './model';
import NanoGPTV2 from './NanoGPTV2';

export default function createModelInstance(config: GPTConfig): Model<ModelForwardAttributes, GPTConfig> {
    console.log(`Creating model instance with config: ${JSON.stringify(config, undefined, 4)}`);
    const modelType = config.modelType || 'GenAI_NanoGPT_v1';
    switch (modelType) {
        case 'GenAI_NanoGPT_v1':
            return new NanoGPT(config as GPTConfigV1);
        case 'GenAI_NanoGPT_v2':
            return new NanoGPTV2(config as GPTConfigV2);
        default:
            throw new Error(`Unsupported model type: ${modelType}`);
    }
}
