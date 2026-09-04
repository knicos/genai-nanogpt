import Model, { ModelForwardAttributes } from '@base/models/model';
import { TrainingOptions } from './types';
import { TokenStore } from './tasks/TokenStore';
import { DatasetMetadata } from '@base/loader/types';
import { ConversationStream } from '@base/data/stream';
import { tokensFromStreams } from './tasks/tokenStream';
import { ITokeniser } from '@base/tokeniser/type';
import { createTrainValidationDatasets, storeFromArray } from './validation';
import { Dataset } from '@tensorflow/tfjs-data';
import { Tensor } from '@tensorflow/tfjs-core';
import { DatasetBuilder } from './DatasetBuilder';

interface PrepareDataResult {
    trainDataset: Dataset<{ xs: Tensor; ys: Tensor }>;
    validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>;
    totalTokens: number;
}

/** Take our training options, model, tokeniser, and tasks, and prepare the training and validation datasets in Tensorflow format. */
export default async function prepareData(
    options: TrainingOptions,
    model: Model<ModelForwardAttributes>,
    tokeniser: ITokeniser,
    tasks: ConversationStream[] | Uint16Array[] | TokenStore,
    datasetId: string,
    validation?: Uint16Array[] | TokenStore,
    datasets?: DatasetMetadata[]
): Promise<PrepareDataResult> {
    const isLoRA = options.loraName || options.loraConfig;
    if (!datasets && !isLoRA) {
        throw new Error('Must specify datasets for non-LoRA training');
    }

    if (datasets) {
        let isConversational = false;
        for (const dataset of datasets) {
            if (dataset.conversational) {
                isConversational = true;
            }
        }

        if (!isLoRA) {
            model.metaData.pretrainingData = datasets.map((d) => ({
                id: d.id,
                name: d.name,
                conversational: d.conversational,
                url: d.url,
            }));
        }

        if (isConversational) {
            model.metaData.mode = 'conversational';
        } else if (model.metaData.mode !== 'conversational') {
            model.metaData.mode = 'completion';
        }
    } else {
        // TODO: Scan for user start token.

        if (model.metaData.mode !== 'conversational') {
            model.metaData.mode = 'completion';
        }
    }

    const maskedLoss = options.maskedLoss ?? options.method.type === 'supervised';

    /*if (this.trainingType === 'sft' && this.trainer instanceof SFTTrainer && tasks instanceof Uint16Array) {
            throw new Error('SFT training requires Task[] input');
        }*/

    let trainingTokens: Uint16Array[] | TokenStore;
    let validationTokens: Uint16Array[] | TokenStore | undefined = validation;

    if (Array.isArray(tasks)) {
        if (tasks[0] instanceof Uint16Array) {
            trainingTokens = tasks as Uint16Array[];
        } else {
            const result = await tokensFromStreams(tasks as ConversationStream[], tokeniser, datasetId, {
                masking: maskedLoss,
                validationSplit: options.validationSplit,
            });
            trainingTokens = result.trainingTokens;
            if (!validation) {
                validationTokens = result.validationTokens;
            }
        }
    } else {
        trainingTokens = tasks as TokenStore;
    }

    console.log('Training tokens', trainingTokens);

    const totalTokens =
        trainingTokens instanceof TokenStore
            ? trainingTokens.getTokenCount()
            : trainingTokens.reduce((sum, shard) => sum + shard.length, 0);

    options.epochSteps = Math.max(1, Math.floor(totalTokens / ((options?.batchSize || 32) * model.config.blockSize)));

    const datasetBuilder = new DatasetBuilder(tokeniser, model.config.blockSize);

    if (validationTokens) {
        const { trainDataset, validationDataset } = await createTrainValidationDatasets(
            trainingTokens,
            validationTokens,
            tokeniser,
            datasetBuilder,
            options?.batchSize || 32
        );

        return { trainDataset, validationDataset, totalTokens };
    } else {
        const tokens =
            trainingTokens instanceof TokenStore
                ? trainingTokens
                : await storeFromArray(trainingTokens as Uint16Array[], tokeniser);
        const trainDataset = (await datasetBuilder.createTextDataset(tokens, options)).dataset;
        return { trainDataset, validationDataset: undefined, totalTokens };
    }

    //this.trainer.updateOptimizer(this.options);
}
