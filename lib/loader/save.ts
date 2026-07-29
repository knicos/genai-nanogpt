import type { ITokeniser } from '@base/tokeniser/type';
import zip from 'jszip';
import CharTokeniser from '../tokeniser/CharTokeniser';
import { Tensor } from '@tensorflow/tfjs-core';
import { save_safetensors } from '../utilities/safetensors';
import { VERSION } from './load';
import { TransformersConfig, TransformersMetadata } from '@base/loader/types';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { AdamWOptimizer } from '@base/training/AdamW';
import { TrainingLogEntry } from '@base/training/types';
import { GPTConfig } from '@base/models/config';

export interface SaveOptions {
    name?: string;
    metadata?: Record<string, unknown>;
    files?: Record<string, unknown>;
    includeOptimizer?: boolean;
    quantize?: 'none' | 'F16';
}

export interface ExtraSaveItems {
    optimizer?: AdamWOptimizer;
    trainingLog?: TrainingLogEntry[];
}

function collapseLog(log: TrainingLogEntry[]): TrainingLogEntry[] {
    // Reduce length by skipping if needed.
    if (log.length > 1000) {
        const step = Math.ceil(log.length / 1000);
        return log.filter((_, index) => index % step === 0 || index === log.length - 1);
    }
    return log;
}

export async function saveModel(
    model: Model<ModelForwardAttributes, GPTConfig>,
    tokeniser: ITokeniser,
    options?: SaveOptions,
    extraItems?: ExtraSaveItems
): Promise<Blob> {
    const weightsMap = new Map<string, Tensor[]>();
    model.weightStore.saveWeights(weightsMap);
    const zipFile = new zip();

    if (extraItems?.optimizer) {
        const optimizerWeights = await extraItems.optimizer.saveMoments();
        zipFile.file('optimizer.safetensors', optimizerWeights as ArrayBuffer, { binary: true });
        zipFile.file('optimizer_config.json', JSON.stringify(extraItems.optimizer.serializeConfig()), {
            binary: false,
        });
    }

    if (extraItems?.trainingLog) {
        zipFile.file('training_log.json', JSON.stringify(collapseLog(extraItems.trainingLog), undefined, 4), {
            binary: false,
        });
    }

    const weights: Record<string, Tensor> = {};
    weightsMap.forEach((tensorList, name) => {
        if (tensorList.length === 1) {
            weights[name] = tensorList[0];
        }
    });

    const weightsBin = await save_safetensors(weights, options?.quantize);
    zipFile.file('model.safetensors', weightsBin as ArrayBuffer, { binary: true });

    const modelType = model.config.modelType;

    let transformersConfig: TransformersConfig;

    if (modelType === 'GenAI_NanoGPT_v1') {
        transformersConfig = {
            model_type: 'GenAI_NanoGPT_v1',
            vocab_size: tokeniser.getVocab().length,
            hidden_size: model.config.nEmbed,
            num_hidden_layers: model.config.nLayer,
            num_attention_heads: model.config.nHead,
            block_size: model.config.blockSize,
            mlpFactor: model.config.mlpFactor,
            useRope: model.config.useRope,
        };
    } else {
        transformersConfig = {
            model_type: 'GenAI_NanoGPT_v2',
            vocab_size: tokeniser.getVocab().length,
            hidden_size: model.config.nEmbed,
            num_hidden_layers: model.config.nLayer,
            num_attention_heads: model.config.nHead,
            block_size: model.config.blockSize,
            mlpFactor: model.config.mlpFactor,
            loraConfig: model.config.loraConfig ? Object.fromEntries(model.config.loraConfig) : undefined,
            loraName: model.config.loraName,
            windowSize: model.config.windowSize,
        };
    }

    zipFile.file('config.json', JSON.stringify(transformersConfig, undefined, 4), {
        binary: false,
    });

    const meta: TransformersMetadata = {
        version: VERSION,
        application: '@genai-fi/nanogpt',
        meta: options?.metadata,
        name: options?.name,
        training: model.metaData?.training || undefined,
        reference: model.metaData?.url || undefined,
        mode: model.metaData?.mode || undefined,
        pretrainingData: model.metaData?.pretrainingData || undefined,
        pretrainingSettings: model.metaData?.pretrainingSettings || undefined,
        generationSettings: model.metaData?.generationSettings || undefined,
        actionLog: model.metaData?.actionLog || undefined,
    };
    zipFile.file('meta.json', JSON.stringify(meta, undefined, 4), {
        binary: false,
    });
    zipFile.file(
        'tokeniser.json',
        JSON.stringify({
            type: tokeniser instanceof CharTokeniser ? 'char' : 'bpe',
            vocab: tokeniser.getVocab(),
            merges: tokeniser.getMerges(),
            datasetID: tokeniser.datasetID,
            id: tokeniser.id,
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
