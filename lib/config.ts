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
}

// Default configuration
export const defaultConfig: GPTConfig = {
    vocabSize: 50304, // GPT-2 vocab size
    blockSize: 1024, // Maximum sequence length
    nLayer: 12, // Number of transformer layers
    nHead: 12, // Number of attention heads
    nEmbed: 768, // Embedding dimension
    dropout: 0.0, // Dropout probability
    biasInLinear: false,
    biasInLayerNorm: false,
};
