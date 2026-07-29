import { ITokeniser, Task, tokensFromTasks } from '@base/main';
import { Tensor } from '@tensorflow/tfjs-core';
import { Dataset } from '@tensorflow/tfjs-data';
import { DatasetBuilder, DatasetState, shuffle } from './DatasetBuilder';

export async function createTrainValidationSplit(
    tasks: Task[] | Uint16Array,
    tokeniser: ITokeniser,
    datasetBuilder: DatasetBuilder,
    batchSize: number,
    validationSplit = 0.1,
    masking?: boolean
): Promise<{
    trainDataset: Dataset<{ xs: Tensor; ys: Tensor }>;
    validationDataset: Dataset<{ xs: Tensor; ys: Tensor }>;
    size: number;
    validationState: DatasetState;
    trainState: DatasetState;
}> {
    const tokens = tasks instanceof Uint16Array ? tasks : await tokensFromTasks(tasks, tokeniser, undefined, masking);
    const allTokens = tokens instanceof Uint16Array ? tokens : tokens.tokens;
    const totalBlocks = Math.ceil(allTokens.length / datasetBuilder.blockSize);
    const mask = tokens instanceof Uint16Array ? undefined : tokens.mask;

    const validationMask = new Set<number>();
    if (validationSplit > 0) {
        const numValidationBlocks = Math.max(1, Math.floor(totalBlocks * validationSplit));

        while (validationMask.size < numValidationBlocks) {
            const blockIndex = Math.floor(Math.random() * totalBlocks);
            validationMask.add(blockIndex);
        }
    }

    const trainIndexes = new Uint32Array(totalBlocks - validationMask.size);
    const validationIndexes = new Uint32Array(validationMask.size);

    let trainIdx = 0;
    let valIdx = 0;
    for (let blockIndex = 0; blockIndex < totalBlocks; blockIndex++) {
        if (validationMask.has(blockIndex)) {
            if (valIdx < validationIndexes.length) {
                validationIndexes[valIdx++] = blockIndex;
            }
        } else {
            if (trainIdx < trainIndexes.length) {
                trainIndexes[trainIdx++] = blockIndex;
            }
        }
    }

    validationMask.clear();

    // Only shuffle validation
    shuffle(validationIndexes);

    const { dataset: trainDataset, state: trainState } = await datasetBuilder.createTextDataset(
        allTokens,
        batchSize,
        trainIndexes,
        mask ? mask : undefined
    );

    const { dataset: validationDataset, state: validationState } = await datasetBuilder.createTextDataset(
        allTokens,
        batchSize,
        validationIndexes
    );

    return { trainDataset, validationDataset, size: allTokens.length, validationState, trainState };
}
