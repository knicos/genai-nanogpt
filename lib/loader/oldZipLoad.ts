import zip from 'jszip';
import { GPTConfig } from '@base/config';
import { BPETokeniser, CharTokeniser, ITokeniser } from '@base/main';
import NanoGPT, { TrainingLogEntry } from '@base/NanoGPTModel';
import { importWeights, ITensorSpec, IWeightManifest } from '@base/utilities/weights';
import { disposeVariables, Tensor } from '@tensorflow/tfjs-core';
import { dummyPassAsync } from '@base/utilities/dummy';

export default async function loadOldModel(zipFile: zip): Promise<{ model: NanoGPT; tokeniser: ITokeniser }> {
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
