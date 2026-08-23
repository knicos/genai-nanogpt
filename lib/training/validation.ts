import { ITokeniser } from '@base/tokeniser/type';
import { Tensor } from '@tensorflow/tfjs-core';
import { Dataset } from '@tensorflow/tfjs-data';
import { DatasetBuilder, DatasetState } from './DatasetBuilder';
import { createTokenStore, TokenStore } from './tasks/TokenStore';

export async function storeFromArray(tokens: Uint16Array[], tokenizer: ITokeniser): Promise<TokenStore> {
    const store = await createTokenStore('training-tokens', tokenizer.id, tokenizer.datasetID ?? '');
    tokens.forEach((tokenArray) => {
        store.appendShard(tokenArray);
    });
    await store.finish();
    return store;
}

export async function createTrainValidationDatasets(
    trainingTokens: Uint16Array[] | TokenStore,
    validationTokens: Uint16Array[] | TokenStore,
    tokeniser: ITokeniser,
    datasetBuilder: DatasetBuilder,
    batchSize: number
): Promise<{
    trainDataset: Dataset<{ xs: Tensor; ys: Tensor }>;
    validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>;
    size: number;
    validationState?: DatasetState;
    trainState: DatasetState;
}> {
    const trainingStore =
        trainingTokens instanceof TokenStore
            ? trainingTokens
            : await storeFromArray(trainingTokens as Uint16Array[], tokeniser);

    const validationStore =
        validationTokens instanceof TokenStore
            ? validationTokens
            : await storeFromArray(validationTokens as Uint16Array[], tokeniser);

    const totalTokens = trainingStore.getTokenCount();

    const { dataset: trainDataset, state: trainState } = await datasetBuilder.createTextDataset(trainingStore, {
        batchSize,
    });

    if (validationStore.getTokenCount() === 0) {
        return {
            trainDataset,
            validationDataset: undefined,
            size: totalTokens,
            validationState: undefined,
            trainState,
        };
    }
    const { dataset: validationDataset, state: validationState } = await datasetBuilder.createTextDataset(
        validationStore,
        {
            batchSize,
            shuffleFirst: true, // Shuffle validation dataset to ensure randomness
        }
    );

    return {
        trainDataset,
        validationDataset,
        size: totalTokens,
        validationState,
        trainState,
    };
}
