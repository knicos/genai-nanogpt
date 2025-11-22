import type { ITokeniser } from '@base/tokeniser/type';
import zip from 'jszip';
import CharTokeniser from '../tokeniser/CharTokeniser';
import { Tensor } from '@tensorflow/tfjs-core';
import { save_safetensors } from '../utilities/safetensors';
import { VERSION } from './load';
import { TransformersConfig } from '@base/loader/loadTransformers';
import Model, { ModelForwardAttributes } from '@base/models/model';

export interface SaveOptions {
    name?: string;
    metadata?: Record<string, unknown>;
    files?: Record<string, unknown>;
}

export async function saveModel(
    model: Model<ModelForwardAttributes>,
    tokeniser: ITokeniser,
    options?: SaveOptions
): Promise<Blob> {
    const weightsMap = new Map<string, Tensor[]>();
    model.saveWeights(weightsMap);
    const zipFile = new zip();

    const weights: Record<string, Tensor> = {};
    weightsMap.forEach((tensorList, name) => {
        if (tensorList.length === 1) {
            weights[name] = tensorList[0];
        }
    });

    const weightsBin = await save_safetensors(weights);
    zipFile.file('model.safetensors', weightsBin as ArrayBuffer, { binary: true });

    const transformersConfig: TransformersConfig = {
        model_type: model.config.modelType || model.getClassName(),
        vocab_size: tokeniser.getVocab().length,
        hidden_size: model.config.nEmbed,
        num_hidden_layers: model.config.nLayer,
        num_attention_heads: model.config.nHead,
        block_size: model.config.blockSize,
        dropout: model.config.dropout,
        biasInLinear: model.config.biasInLinear,
        biasInLayerNorm: model.config.biasInLayerNorm,
        mlpFactor: model.config.mlpFactor,
        useRope: model.config.useRope,
    };

    zipFile.file('config.json', JSON.stringify(transformersConfig, undefined, 4), {
        binary: false,
    });

    zipFile.file(
        'meta.json',
        JSON.stringify(
            {
                version: VERSION,
                application: '@genai-fi/nanogpt',
                meta: options?.metadata,
                name: options?.name,
                training: model.trainingState || undefined,
            },
            undefined,
            4
        ),
        {
            binary: false,
        }
    );
    zipFile.file(
        'tokeniser.json',
        JSON.stringify({
            type: tokeniser instanceof CharTokeniser ? 'char' : 'bpe',
            vocab: tokeniser.getVocab(),
            merges: await tokeniser.getMerges(),
        }),
        {
            binary: false,
        }
    );

    if (options?.files) {
        for (const [fileName, content] of Object.entries(options.files)) {
            zipFile.file(fileName, JSON.stringify(content), { binary: false });
        }
    }

    return zipFile.generateAsync({ type: 'blob' });
}
