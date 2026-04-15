import { ITokeniser, Task, tokensFromTasks } from '@base/main';
import { Tensor } from '@tensorflow/tfjs-core';
import { Dataset } from '@tensorflow/tfjs-data';
import { DatasetBuilder, DatasetState, PAGE_FACTOR, shuffle } from './DatasetBuilder';

export async function createTrainValidationSplit(
    tasks: Task[] | Uint16Array,
    tokeniser: ITokeniser,
    datasetBuilder: DatasetBuilder,
    batchSize: number,
    validationSplit = 0.1
): Promise<{
    trainDataset: Dataset<{ xs: Tensor; ys: Tensor }>;
    validationDataset: Dataset<{ xs: Tensor; ys: Tensor }>;
    size: number;
    validationState: DatasetState;
    trainState: DatasetState;
}> {
    const allTokens = tasks instanceof Uint16Array ? tasks : await tokensFromTasks(tasks, tokeniser);

    const validationMask = new Set<number>();
    if (validationSplit > 0) {
        const totalPages = Math.floor(allTokens.length / (datasetBuilder.blockSize * PAGE_FACTOR));
        const numValidationPages = Math.max(1, Math.floor(totalPages * validationSplit));

        while (validationMask.size < numValidationPages) {
            const pageIndex = Math.floor(Math.random() * totalPages);
            validationMask.add(pageIndex);
        }
    }

    const trainIndexes = new Uint32Array(
        allTokens.length - validationMask.size * datasetBuilder.blockSize * PAGE_FACTOR
    );
    const validationIndexes = new Uint32Array(validationMask.size * datasetBuilder.blockSize * PAGE_FACTOR);

    let trainIdx = 0;
    let valIdx = 0;
    for (let i = 0; i < allTokens.length; i++) {
        const pageIndex = Math.floor(i / (datasetBuilder.blockSize * PAGE_FACTOR));
        if (validationMask.has(pageIndex)) {
            if (valIdx < validationIndexes.length) {
                validationIndexes[valIdx++] = i;
            }
        } else {
            if (trainIdx < trainIndexes.length) {
                trainIndexes[trainIdx++] = i;
            }
        }
    }

    const { dataset: trainDataset, state: trainState } = await datasetBuilder.createTextDataset(
        allTokens,
        batchSize,
        shuffle(trainIndexes)
    );

    const { dataset: validationDataset, state: validationState } = await datasetBuilder.createTextDataset(
        allTokens,
        batchSize,
        shuffle(validationIndexes)
    );

    return { trainDataset, validationDataset, size: allTokens.length, validationState, trainState };
}
