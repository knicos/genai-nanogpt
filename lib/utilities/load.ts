import zip from 'jszip';
import { importWeights, ITensorSpec, IWeightManifest } from './weights';
import CharTokeniser from '../tokeniser/CharTokeniser';
import NanoGPT, { TrainingLogEntry } from '../NanoGPTModel';
import type { ITokeniser } from '@base/tokeniser/type';
import { GPTConfig } from '../config';
import { dummyPassAsync } from './dummy';
import { disposeVariables, Tensor } from '@tensorflow/tfjs-core';
import BPETokeniser from '../tokeniser/bpe';

async function loadURL(url: string): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
    }
    return response.arrayBuffer();
}

export async function loadModel(data: Blob | Buffer | string): Promise<{ model: NanoGPT; tokeniser: ITokeniser }> {
    const blob = typeof data === 'string' ? await loadURL(data) : data;

    const zipFile = await zip.loadAsync(blob);
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
