export interface LoRAConfig {
    rank: number;
    alpha: number;
    variables: string[];
}

// Configuration for the nanoGPT model
export interface GPTConfig {
    modelType?: string;
    vocabSize: number;
    blockSize: number;
    nLayer: number;
    nHead: number;
    nEmbed: number;
    biasInLinear: boolean;
    biasInLayerNorm: boolean;
    mlpFactor: number;
    useRope: boolean;
    loraConfig?: LoRAConfig;
    noRMSLearnables?: boolean;
}

// Default configuration
export const defaultConfig: GPTConfig = {
    modelType: 'GenAI_NanoGPT_v1',
    vocabSize: 2000,
    blockSize: 128, // Maximum sequence length
    nLayer: 6, // Number of transformer layers
    nHead: 4, // Number of attention heads
    nEmbed: 256, // Embedding dimension
    biasInLinear: false,
    biasInLayerNorm: false,
    mlpFactor: 4,
    useRope: true, // Use Rotary Position Embeddings
    noRMSLearnables: false, // Whether to make RMSNorm parameters learnable
};
