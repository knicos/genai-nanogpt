import { GPTConfig } from '@base/models/config';
import { ITokeniser } from '@base/tokeniser/type';
import CharTokeniser from '@base/tokeniser/CharTokeniser';
import BPETokeniser from '@base/tokeniser/bpe';
import { load_safetensors } from '@base/utilities/safetensors';
import { disposeVariables, Tensor } from '@tensorflow/tfjs-core';
import { dummyPassAsync } from '@base/utilities/dummy';
import createModelInstance from '@base/models/factory';
import Model, { ModelForwardAttributes } from '@base/models/model';

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
    [key: string]: unknown;
}

export default async function loadTransformers(
    config: TransformersConfig,
    tokeniser: TransformersTokeniser,
    metadata: TransformersMetadata,
    weightData: ArrayBuffer
): Promise<{ model: Model<ModelForwardAttributes>; tokeniser: ITokeniser; name?: string }> {
    const modelConfig: GPTConfig = {
        modelType: config.model_type || 'GenAI_NanoGPT_v1',
        vocabSize: config.vocab_size,
        blockSize: config.block_size,
        nLayer: config.num_hidden_layers,
        nHead: config.num_attention_heads,
        nEmbed: config.hidden_size,
        dropout: config.dropout,
        biasInLinear: config.biasInLinear,
        biasInLayerNorm: config.biasInLayerNorm,
        mlpFactor: config.mlpFactor,
        useRope: config.useRope,
    };

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

    await dummyPassAsync(model); // Initialize the model to set up weights and caches
    model.loadWeights(weightsMap);

    return { model, tokeniser: tokeniserInstance, name: metadata.name };
}
