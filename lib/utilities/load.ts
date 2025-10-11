import zip from 'jszip';
import { importWeights, ITensorSpec, IWeightManifest } from './weights';
import CharTokeniser from '../tokeniser/CharTokeniser';
import NanoGPT, { TrainingLogEntry } from '../NanoGPTModel';
import type { ITokeniser } from '@base/tokeniser/type';
import { GPTConfig } from '../config';
import { dummyPassAsync } from './dummy';
import { disposeVariables, Tensor } from '@tensorflow/tfjs-core';
import BPETokeniser from '../tokeniser/bpe';
import { load_safetensors } from './safetensors';

export const VERSION = 2;

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

export interface Metadata {
    version: string;
    application: string;
    name?: string;
}

async function loadURL(url: string): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
    }
    return response.arrayBuffer();
}

export async function loadOldModel(zipFile: zip): Promise<{ model: NanoGPT; tokeniser: ITokeniser }> {
    const manifests = new Map<string, IWeightManifest>();

    const manifestFile = await zipFile.file('manifest.json')?.async('string');
    if (!manifestFile) {
        throw new Error('Manifest file not found in the zip archive');
    }
    const manifest = JSON.parse(manifestFile) as {
        weightSpec: Record<string, ITensorSpec[]>;
        config: GPTConfig;
        vocab: string[];
    };
    for (const [name, specs] of Object.entries(manifest.weightSpec)) {
        manifests.set(name, { spec: specs, data: new Float32Array() });
    }

    const tokeniserFile = await zipFile.file('tokeniser.json')?.async('string');
    if (!tokeniserFile) {
        throw new Error('Tokeniser file not found in the zip archive');
    }
    const tokeniserData = JSON.parse(tokeniserFile) as {
        type: 'char' | 'bpe';
        vocab: string[];
        merges: [string, string][];
    };

    const tokeniserType = tokeniserData.type ?? 'char';

    const tokeniser =
        tokeniserType === 'char'
            ? new CharTokeniser(tokeniserData.vocab)
            : new BPETokeniser(tokeniserData.vocab, tokeniserData.merges);

    const weights = new Map<string, Tensor[]>();

    for (const fileName of Object.keys(zipFile.files)) {
        if (fileName.endsWith('.bin')) {
            const name = fileName.replace('.bin', '');
            const data = await zipFile.file(fileName)!.async('arraybuffer');
            const floatData = new Float32Array(data);
            const entry = manifests.get(name) || { spec: [], data: new Float32Array() };
            entry.data = floatData;
            manifests.set(name, entry);

            const tensors = await importWeights(entry);
            weights.set(name, tensors);
        }
    }

    // Force existing variables to be removed
    disposeVariables();

    const model = new NanoGPT(manifest.config);

    await dummyPassAsync(model); // Initialize the model to set up weights and caches
    model.loadWeights(weights);

    const logFile = await zipFile.file('log.json')?.async('string');
    if (logFile) {
        try {
            const logData: TrainingLogEntry[] = JSON.parse(logFile);
            model.log = logData;
        } catch (error) {
            console.error('Error parsing training log:', error);
            throw new Error(`Failed to parse training log: ${error}`);
        }
    }

    return { model, tokeniser };
}

export async function loadModel(
    data: Blob | Buffer | string
): Promise<{ model: NanoGPT; tokeniser: ITokeniser; name?: string }> {
    const blob = typeof data === 'string' ? await loadURL(data) : data;

    const zipFile = await zip.loadAsync(blob);

    if (zipFile.file('manifest.json')) {
        return loadOldModel(zipFile);
    } else {
        const configFile = await zipFile.file('config.json')?.async('string');
        if (!configFile) {
            throw new Error('Config file not found in the zip archive');
        }
        const config = JSON.parse(configFile) as TransformersConfig;

        const modelConfig: GPTConfig = {
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

        const tokeniserFile = await zipFile.file('tokeniser.json')?.async('string');
        if (!tokeniserFile) {
            throw new Error('Tokeniser file not found in the zip archive');
        }
        const tokeniserData = JSON.parse(tokeniserFile) as {
            type: 'char' | 'bpe';
            vocab: string[];
            merges: [string, string][];
        };

        const tokeniserType = tokeniserData.type ?? 'char';

        const tokeniser =
            tokeniserType === 'char'
                ? new CharTokeniser(tokeniserData.vocab)
                : new BPETokeniser(tokeniserData.vocab, tokeniserData.merges);

        const weights = await load_safetensors(await zipFile.file('model.safetensors')!.async('arraybuffer'));
        const weightsMap = new Map<string, Tensor[]>();
        for (const [key, value] of Object.entries(weights)) {
            weightsMap.set(key, [value]);
        }

        // Force existing variables to be removed
        disposeVariables();

        const model = new NanoGPT(modelConfig);

        await dummyPassAsync(model); // Initialize the model to set up weights and caches
        model.loadWeights(weightsMap);

        const metaFile = await zipFile.file('meta.json')?.async('string');
        let name: string | undefined = undefined;
        if (metaFile) {
            try {
                const metaData = JSON.parse(metaFile);
                if (metaData.name) {
                    name = metaData.name;
                }
            } catch (error) {
                console.error('Error parsing meta file:', error);
            }
        }

        const logFile = await zipFile.file('log.json')?.async('string');
        if (logFile) {
            try {
                const logData: TrainingLogEntry[] = JSON.parse(logFile);
                model.log = logData;
            } catch (error) {
                console.error('Error parsing training log:', error);
                throw new Error(`Failed to parse training log: ${error}`);
            }
        }

        return { model, tokeniser, name };
    }
}
