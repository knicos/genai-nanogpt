import zip from 'jszip';
import loadTransformers from './loadTransformers';
import { LoadResult, TransformersConfig, TransformersMetadata, TransformersTokeniser } from './types';
import { AdamWOptimizer } from '@base/training/AdamW';
import { TrainingLogEntry } from '@base/training/types';

export default async function loadZipFile(zipFile: zip, metaData: TransformersMetadata): Promise<LoadResult> {
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

    const optimizerConfigFile = await zipFile.file('optimizer_config.json')?.async('string');
    let optimizer: AdamWOptimizer | undefined = undefined;
    if (optimizerConfigFile) {
        const optimizerConfig = JSON.parse(optimizerConfigFile);
        const optimizerWeightsData = await zipFile.file('optimizer.safetensors')?.async('arraybuffer');
        if (!optimizerWeightsData) {
            throw new Error('Optimizer weights not found in the zip archive');
        }
        optimizer = new AdamWOptimizer(optimizerConfig);
        await optimizer.loadMoments(optimizerWeightsData);
    }

    const logFile = await zipFile.file('training_log.json')?.async('string');
    let trainingLog: TrainingLogEntry[] | undefined = undefined;
    if (logFile) {
        trainingLog = JSON.parse(logFile) as TrainingLogEntry[];
    }

    return {
        ...(await loadTransformers(config, tokeniserData, metaData, weightData)),
        optimizer,
        log: trainingLog,
    };
}
