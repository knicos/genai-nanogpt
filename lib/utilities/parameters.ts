import { GPTConfig } from '@base/models/config';
import { swigluHidden } from '@base/layers/MLP';

const BYTES_PER_PARAMETER = 4; // Assuming float32

function estimateMLPParameters(config: GPTConfig): number {
    let mlpParams: number;
    switch (config.modelType) {
        case 'GenAI_NanoGPT_v1':
        case 'GenAI_NanoGPT_v2':
            mlpParams = swigluHidden(config.nEmbed, config.mlpFactor) * config.nEmbed * 2;
            break;
        case 'llama':
        case 'qwen3':
            // SwiGLU gated MLP
            mlpParams = Math.max(256, Math.ceil((config.nEmbed * (8 / 3)) / 256) * 256) * config.nEmbed * 3;
            break;
        default:
            mlpParams = 0;
    }
    return mlpParams;
}

export function estimateLayerParameters(config: GPTConfig): number {
    const attentionParams = 4 * config.nEmbed * config.nEmbed; // qkv + proj
    const mlpParams = estimateMLPParameters(config);

    return attentionParams + mlpParams;
}

export function estimateParameterCount(config: GPTConfig): number {
    const embeddingParams =
        config.modelType === 'llama' ? config.vocabSize * config.nEmbed * 2 : config.vocabSize * config.nEmbed;
    const attentionParams = config.nLayer * (4 * config.nEmbed * config.nEmbed); // qkv + proj

    const mlpParams = estimateMLPParameters(config) * config.nLayer;

    return embeddingParams + attentionParams + mlpParams;
}

export function estimateMemoryUsage(config: GPTConfig): number {
    const numParams = estimateParameterCount(config);
    return numParams * BYTES_PER_PARAMETER;
}

export function estimateTrainingMemoryUsage(config: GPTConfig, batchSize: number): number {
    const modelMemory = estimateMemoryUsage(config);
    const activationMemory = modelMemory * 2; // Activations
    const optimizerMemory = modelMemory * 2; // Optimizer states
    const batchMemory = batchSize * config.blockSize * config.nEmbed * BYTES_PER_PARAMETER;

    return modelMemory + activationMemory + optimizerMemory + batchMemory;
}

export function estimateResources(config: GPTConfig, batchSize: number) {
    const numParams = estimateParameterCount(config);
    const modelMemoryMB = estimateMemoryUsage(config) / (1024 * 1024);
    const trainingMemoryMB = estimateTrainingMemoryUsage(config, batchSize) / (1024 * 1024);

    return {
        numParams,
        modelMemoryMB,
        trainingMemoryMB,
    };
}

export function validateConfig(config: GPTConfig) {
    if (config.nEmbed % config.nHead !== 0) {
        throw new Error('nEmbed_divisible_nHead');
    }
    if (config.blockSize <= 0) {
        throw new Error('blockSize_positive');
    }
    if (config.vocabSize <= 0) {
        throw new Error('vocabSize_positive');
    }
    if (config.nLayer <= 0) {
        throw new Error('nLayer_positive');
    }
    if (config.mlpFactor <= 0) {
        throw new Error('mlpFactor_positive');
    }

    const headDim = config.nEmbed / config.nHead;
    if (headDim % 2 !== 0) {
        throw new Error('headDim_even');
    }
    if (!Number.isInteger(config.nEmbed)) {
        throw new Error('nEmbed_integer');
    }
    if (!Number.isInteger(config.nHead)) {
        throw new Error('nHead_integer');
    }
    if (!Number.isInteger(config.nLayer)) {
        throw new Error('nLayer_integer');
    }
    if (!Number.isInteger(config.blockSize)) {
        throw new Error('blockSize_integer');
    }
    if (!Number.isInteger(config.vocabSize)) {
        throw new Error('vocabSize_integer');
    }
}
