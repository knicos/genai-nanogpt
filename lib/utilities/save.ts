import NanoGPT from '@base/NanoGPTModel';
import { ITokeniser } from '@base/tokeniser/type';
import zip from 'jszip';
import { exportWeights, ITensorSpec } from './weights';

export async function saveModel(model: NanoGPT, tokeniser: ITokeniser): Promise<Blob> {
    const weights = model.saveWeights();
    const zipFile = new zip();

    const spec: Record<string, ITensorSpec[]> = {};

    for (const [name, tensorList] of weights) {
        const data = await exportWeights(tensorList);
        spec[name] = data.spec;
        zipFile.file(`${name}.bin`, data.data.buffer, { binary: true });
    }
    zipFile.file('manifest.json', JSON.stringify({ weightSpec: spec, config: model.config }), {
        binary: false,
    });
    zipFile.file(
        'tokeniser.json',
        JSON.stringify({ vocab: tokeniser.getVocab(), merges: await tokeniser.getMerges() }),
        {
            binary: false,
        }
    );
    zipFile.file('log.json', JSON.stringify(model.log), { binary: false });
    return zipFile.generateAsync({ type: 'blob' });
}
