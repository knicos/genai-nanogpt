import { GPTConfig } from '@base/models/config';
import { ITokeniser } from '@base/tokeniser/type';
import CharTokeniser from '@base/tokeniser/CharTokeniser';
import BPETokeniser from '@base/tokeniser/bpe';
import { load_safetensors } from '@base/utilities/safetensors';
import { disposeVariables, Tensor } from '@tensorflow/tfjs-core';
import { dummyPassAsync } from '@base/utilities/dummy';
import createModelInstance from '@base/models/factory';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { TransformersConfig, TransformersMetadata, TransformersTokeniser } from './types';

export function mapTransformersConfigToGPTConfig(config: TransformersConfig): GPTConfig {
    const modelConfig: GPTConfig = {
        modelType: config.model_type || 'GenAI_NanoGPT_v1',
        vocabSize: config.vocab_size,
        blockSize: config.block_size,
        nLayer: config.num_hidden_layers,
        nHead: config.num_attention_heads,
        nEmbed: config.hidden_size,
        biasInLinear: config.biasInLinear,
        biasInLayerNorm: config.biasInLayerNorm,
        mlpFactor: config.mlpFactor,
        useRope: config.useRope,
        loraConfig: config.loraConfig,
        noRMSLearnables: config.noRMSLearnables,
    };

    return modelConfig;
}

export default async function loadTransformers(
    config: TransformersConfig,
    tokeniser: TransformersTokeniser,
    metadata: TransformersMetadata,
    weightData: ArrayBuffer
): Promise<{ model: Model<ModelForwardAttributes>; tokeniser: ITokeniser; metaData: TransformersMetadata }> {
    const modelConfig = mapTransformersConfigToGPTConfig(config);

    const tokeniserType = tokeniser.type ?? 'char';

    const tokeniserInstance =
        tokeniserType === 'char'
            ? new CharTokeniser(tokeniser.vocab)
            : new BPETokeniser(tokeniser.vocab, tokeniser.merges);

    const weights = await load_safetensors(weightData);
    const weightsMap = new Map<string, Tensor[]>();
    for (const [key, value] of Object.entries(weights)) {
        weightsMap.set(key, [value]);
    }

    // Force existing variables to be removed
    disposeVariables();

    const model = createModelInstance(modelConfig);
    model.metaData = metadata;

    await dummyPassAsync(model); // Initialize the model to set up weights and caches
    model.weightStore.loadWeights(weightsMap, metadata.url ? true : false); // If loaded from URL, treat as reference to avoid saving again

    return { model, tokeniser: tokeniserInstance, metaData: metadata };
}
