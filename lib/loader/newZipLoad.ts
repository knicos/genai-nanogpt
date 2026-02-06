import { ITokeniser } from '@base/main';
import zip from 'jszip';
import loadTransformers from './loadTransformers';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { TransformersConfig, TransformersMetadata, TransformersTokeniser } from './types';

export default async function loadZipFile(
    zipFile: zip,
    metaData: TransformersMetadata
): Promise<{ model: Model<ModelForwardAttributes>; tokeniser: ITokeniser; metaData: TransformersMetadata }> {
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

    const weightData = await zipFile.file('model.safetensors')?.async('arraybuffer');
    if (!weightData) {
        throw new Error('Model weights not found in the zip archive');
    }

    return loadTransformers(config, tokeniserData, metaData, weightData);
}
