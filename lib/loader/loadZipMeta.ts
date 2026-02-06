import zip from 'jszip';
import { TransformersMetadata } from './types';

export async function loadZipMeta(zipFile: zip): Promise<TransformersMetadata> {
    const metaFile = await zipFile.file('meta.json')?.async('string');
    let metaData: TransformersMetadata = { version: 0, application: '' };
    if (metaFile) {
        try {
            metaData = JSON.parse(metaFile) as TransformersMetadata;
        } catch (error) {
            console.error(error);
            throw new Error('Failed to parse meta.json in the zip archive');
        }
    } else {
        console.warn('meta.json not found in the zip archive, using default metadata');
    }
    return metaData;
}
