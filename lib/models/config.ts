export interface LoRAConfig {
    rank: number;
    alpha: number;
    variables: string[];
}

// Configuration for the nanoGPT model
export interface GPTConfigBase {
    modelType?: string;
    vocabSize: number;
    blockSize: number;
    nLayer: number;
    nHead: number;
    nEmbed: number;
    mlpFactor: number;
    loraConfig?: Map<string, LoRAConfig>;
    loraName?: string;
}

export interface GPTConfigV1 extends GPTConfigBase {
    modelType: 'GenAI_NanoGPT_v1';
    useRope: boolean;
}

export interface GPTConfigV2 extends GPTConfigBase {
    modelType: 'GenAI_NanoGPT_v2';
    windowSize?: string; // S or L for each layer, e.g. 'SSLLS' for 5 layers
}

export type GPTConfig = GPTConfigV1 | GPTConfigV2;

// Default configuration
export const defaultConfig: GPTConfig = {
    modelType: 'GenAI_NanoGPT_v2',
    vocabSize: 2000,
    blockSize: 128, // Maximum sequence length
    nLayer: 6, // Number of transformer layers
    nHead: 4, // Number of attention heads
    nEmbed: 256, // Embedding dimension
    mlpFactor: 4,
};

function assertNumber(obj: Record<string, unknown>, key: string) {
    if (typeof obj[key] !== 'number' || Number.isNaN(obj[key])) {
        throw new Error(`Invalid config: "${key}" must be a number.`);
    }
}

export function validateConfig(config: unknown): asserts config is GPTConfig {
    const isObject = (value: unknown): value is Record<string, unknown> =>
        typeof value === 'object' && value !== null && !Array.isArray(value);

    if (!isObject(config)) {
        throw new Error('Invalid config: expected an object.');
    }

    // Fields from GPTConfigBase
    assertNumber(config, 'vocabSize');
    assertNumber(config, 'blockSize');
    assertNumber(config, 'nLayer');
    assertNumber(config, 'nHead');
    assertNumber(config, 'nEmbed');
    assertNumber(config, 'mlpFactor');

    // Optional loraConfig
    if (config.loraConfig !== undefined) {
        if (!isObject(config.loraConfig)) {
            throw new Error('Invalid config: "loraConfig" must be an object.');
        }

        const values = Object.values(config.loraConfig);
        if (!values.every((v) => isObject(v))) {
            throw new Error('Invalid config: each entry in "loraConfig" must be an object.');
        }
        if (!values.every((v) => 'rank' in v && 'alpha' in v && 'variables' in v)) {
            throw new Error('Invalid config: each LoRA config must have "rank", "alpha", and "variables" fields.');
        }
        values.forEach((v) => {
            assertNumber(v, 'rank');
            assertNumber(v, 'alpha');
            if (!Array.isArray(v.variables) || !v.variables.every((x) => typeof x === 'string')) {
                throw new Error('Invalid config: "variables" must be a string array.');
            }
        });
    }

    // Discriminated union checks
    if (config.modelType === 'GenAI_NanoGPT_v1') {
        if (typeof config.useRope !== 'boolean') {
            throw new Error('Invalid config: "useRope" must be a boolean for GenAI_NanoGPT_v1.');
        }
        return;
    }

    if (config.modelType === 'GenAI_NanoGPT_v2') {
        if (config.windowSize !== undefined && typeof config.windowSize !== 'string') {
            throw new Error('Invalid config: "windowSize" must be a string for GenAI_NanoGPT_v2.');
        }
        return;
    }

    throw new Error('invalid_model_type');
}
