import zip from 'jszip';
import type { ITokeniser } from '@base/tokeniser/type';
import loadOldModel from './oldZipLoad';
import loadZipFile from './newZipLoad';
import loadHuggingFace from './loadHF';
import Model, { ModelForwardAttributes } from '@base/models/model';

export const VERSION = 2;

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

export async function loadModel(
    data: Blob | Buffer | string
): Promise<{ model: Model<ModelForwardAttributes>; tokeniser: ITokeniser; name?: string }> {
    if (typeof data === 'string') {
        if (data.startsWith('http://') || data.startsWith('https://')) {
            const arrayBuffer = await loadURL(data);
            const zipFile = await zip.loadAsync(arrayBuffer);

            if (zipFile.file('manifest.json')) {
                return loadOldModel(zipFile);
            } else {
                return loadZipFile(zipFile);
            }
        } else {
            return loadHuggingFace(data);
        }
    } else {
        const zipFile = await zip.loadAsync(data);

        if (zipFile.file('manifest.json')) {
            return loadOldModel(zipFile);
        } else {
            return loadZipFile(zipFile);
        }
    }
}
