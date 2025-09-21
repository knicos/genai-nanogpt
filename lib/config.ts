// Configuration for the nanoGPT model
export interface GPTConfig {
    vocabSize: number;
    blockSize: number;
    nLayer: number;
    nHead: number;
    nEmbed: number;
    dropout: number;
    biasInLinear: boolean;
    biasInLayerNorm: boolean;
    mlpFactor: number;
    useRope: boolean;
}

// Default configuration
export const defaultConfig: GPTConfig = {
    vocabSize: 2000,
    blockSize: 128, // Maximum sequence length
    nLayer: 6, // Number of transformer layers
    nHead: 4, // Number of attention heads
    nEmbed: 256, // Embedding dimension
    dropout: 0.1, // Dropout probability
    biasInLinear: false,
    biasInLayerNorm: false,
    mlpFactor: 4,
    useRope: true, // Use Rotary Position Embeddings
};
