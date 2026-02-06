import zip from 'jszip';
import type { ITokeniser } from '@base/tokeniser/type';
import loadOldModel from './oldZipLoad';
import loadZipFile from './newZipLoad';
import loadHuggingFace from './loadHF';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { loadZipMeta } from './loadZipMeta';
import { load_safetensors } from '@base/utilities/safetensors';
import { Tensor } from '@tensorflow/tfjs-core';
import { TransformersMetadata } from './types';

export const VERSION = 2;

async function loadURL(url: string): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
    }
    return response.arrayBuffer();
}

async function mergeWeights(zipFile: zip, model: Model<ModelForwardAttributes>, reference: boolean): Promise<void> {
    const file = zipFile.file('model.safetensors');
    if (!file) {
        return;
    }
    const weightData = await file.async('arraybuffer');
    const weights = await load_safetensors(weightData);
    const weightsMap = new Map<string, Tensor[]>();
    for (const [key, value] of Object.entries(weights)) {
        weightsMap.set(key, [value]);
    }
    model.weightStore.loadWeights(weightsMap, reference);
}

async function zipLoadCommon(
    zipFile: zip,
    metaData: TransformersMetadata
): Promise<{ model: Model<ModelForwardAttributes>; tokeniser: ITokeniser; metaData: TransformersMetadata }> {
    // This model refers to another one, so load it first.
    if (metaData.reference) {
        const refModel = await loadModel(metaData.reference);

        // If this model has a URL then load the weights in reference mode to prevent resaving.
        await mergeWeights(zipFile, refModel.model, metaData.url ? true : false);

        // TODO: Validate that configs are compatible between the reference and the new weights.

        return {
            ...refModel,
            metaData: {
                ...refModel.metaData,
                ...metaData,
            },
        };
    } else {
        console.log('Loading as reference', metaData.reference, metaData.url);

        if (zipFile.file('manifest.json')) {
            return loadOldModel(zipFile, metaData);
        } else {
            return loadZipFile(zipFile, metaData);
        }
    }
}

export interface LoadModelOptions {
    sourceURL?: string;
}

export async function loadModel(
    data: Blob | Buffer | string,
    options?: LoadModelOptions
): Promise<{ model: Model<ModelForwardAttributes>; tokeniser: ITokeniser; metaData: TransformersMetadata }> {
    if (typeof data === 'string') {
        if (data.startsWith('http://') || data.startsWith('https://')) {
            const arrayBuffer = await loadURL(data);
            const zipFile = await zip.loadAsync(arrayBuffer);

            const metaData = await loadZipMeta(zipFile);
            // Treat as a reference model if saved again.
            metaData.url = data;
            return zipLoadCommon(zipFile, metaData);
        } else {
            return loadHuggingFace(data);
        }
    } else {
        const zipFile = await zip.loadAsync(data);
        const metaData = await loadZipMeta(zipFile);
        // Cannot be used as a reference if saved again, unless URL is provided in options.
        metaData.url = options?.sourceURL || undefined;
        return zipLoadCommon(zipFile, metaData);
    }
}
