import NanoGPT from '@base/NanoGPTModel';
import type { ITokeniser } from '@base/tokeniser/type';
import zip from 'jszip';
import CharTokeniser from '../tokeniser/CharTokeniser';
import { Tensor } from '@tensorflow/tfjs-core';
import { save_safetensors } from './safetensors';
import { TransformersConfig, VERSION } from './load';

export interface SaveOptions {
    includeLog?: boolean;
    name?: string;
    metadata?: Record<string, unknown>;
    files?: Record<string, unknown>;
}

export async function saveModel(model: NanoGPT, tokeniser: ITokeniser, options?: SaveOptions): Promise<Blob> {
    const includeLog = options?.includeLog ?? true;
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
        model_type: 'GenAI_NanoGPT_1',
        vocab_size: tokeniser.getVocab().length,
        hidden_size: model.config.gpt.nEmbed,
        num_hidden_layers: model.config.gpt.nLayer,
        num_attention_heads: model.config.gpt.nHead,
        block_size: model.config.gpt.blockSize,
        dropout: model.config.gpt.dropout,
        biasInLinear: model.config.gpt.biasInLinear,
        biasInLayerNorm: model.config.gpt.biasInLayerNorm,
        mlpFactor: model.config.gpt.mlpFactor,
        useRope: model.config.gpt.useRope,
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

    if (includeLog) {
        zipFile.file('log.json', JSON.stringify(model.log), { binary: false });
    }

    if (options?.files) {
        for (const [fileName, content] of Object.entries(options.files)) {
            zipFile.file(fileName, JSON.stringify(content), { binary: false });
        }
    }

    return zipFile.generateAsync({ type: 'blob' });
}
