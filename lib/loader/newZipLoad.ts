import { ITokeniser } from '@base/main';
import NanoGPT from '@base/NanoGPTModel';
import zip from 'jszip';
import loadTransformers, { TransformersConfig, TransformersMetadata, TransformersTokeniser } from './loadTransformers';

export default async function loadZipFile(
    zipFile: zip
): Promise<{ model: NanoGPT; tokeniser: ITokeniser; name?: string }> {
    const configFile = await zipFile.file('config.json')?.async('string');
    if (!configFile) {
        throw new Error('Config file not found in the zip archive');
    }
    const config = JSON.parse(configFile) as TransformersConfig;

    const tokeniserFile = await zipFile.file('tokeniser.json')?.async('string');
    if (!tokeniserFile) {
        throw new Error('Tokeniser file not found in the zip archive');
    }
    const tokeniserData = JSON.parse(tokeniserFile) as TransformersTokeniser;

    const weightData = await zipFile.file('model.safetensors')!.async('arraybuffer');

    const metaFile = await zipFile.file('meta.json')?.async('string');
    let metaData: TransformersMetadata = { version: 0, application: '' };
    if (metaFile) {
        try {
            metaData = JSON.parse(metaFile) as TransformersMetadata;
        } catch (error) {
            console.error('Error parsing meta file:', error);
        }
    }

    return loadTransformers(config, tokeniserData, metaData, weightData);
}
