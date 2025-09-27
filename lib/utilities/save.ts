import NanoGPT from '@base/NanoGPTModel';
import type { ITokeniser } from '@base/tokeniser/type';
import zip from 'jszip';
import { exportWeights, ITensorSpec } from './weights';
import CharTokeniser from '../tokeniser/CharTokeniser';
import { Tensor } from '@tensorflow/tfjs-core';

const VERSION = '1.0.0';

export interface SaveOptions {
    includeLog?: boolean;
    name?: string;
    metadata?: Record<string, unknown>;
    files?: Record<string, unknown>;
}

export async function saveModel(model: NanoGPT, tokeniser: ITokeniser, options?: SaveOptions): Promise<Blob> {
    const includeLog = options?.includeLog ?? true;
    const weights = new Map<string, Tensor[]>();
    model.saveWeights(weights);
    const zipFile = new zip();

    const spec: Record<string, ITensorSpec[]> = {};

    for (const [name, tensorList] of weights) {
        const data = await exportWeights(tensorList);
        spec[name] = data.spec;
        zipFile.file(`${name}.bin`, data.data.buffer as ArrayBuffer, { binary: true });
    }
    zipFile.file(
        'manifest.json',
        JSON.stringify({
            weightSpec: spec,
            config: model.config.gpt,
            version: VERSION,
            application: '@genai-fi/nanogpt',
            meta: options?.metadata,
            name: options?.name,
        }),
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
