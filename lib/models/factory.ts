import { GPTConfig } from '@base/models/config';
import NanoGPT from './NanoGPTV1';
import Model, { ModelForwardAttributes } from './model';

export default function createModelInstance(config: GPTConfig): Model<ModelForwardAttributes> {
    const modelType = config.modelType || 'GenAI_NanoGPT_v1';
    switch (modelType) {
        case 'GenAI_NanoGPT_1':
        case 'GenAI_NanoGPT_v1':
            return new NanoGPT(config);
        default:
            throw new Error(`Unsupported model type: ${modelType}`);
    }
}
