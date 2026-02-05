import { ITokeniser, Task, tokensFromTasks } from '@base/main';
import { Tensor } from '@tensorflow/tfjs-core';
import { Dataset } from '@tensorflow/tfjs-data';
import { DatasetBuilder, PAGE_FACTOR } from './DatasetBuilder';

export async function createTrainValidationSplit(
    tasks: Task[] | Uint16Array,
    tokeniser: ITokeniser,
    datasetBuilder: DatasetBuilder,
    batchSize: number,
    validationSplit: number = 0.1
): Promise<{
    trainDataset: Dataset<{ xs: Tensor; ys: Tensor }>;
    validationDataset: Dataset<{ xs: Tensor; ys: Tensor }>;
    size: number;
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

    const trainDataset = await datasetBuilder.createTextDataset(allTokens, batchSize, validationMask, false);
    const validationDataset = await datasetBuilder.createTextDataset(allTokens, batchSize, validationMask, true);

    return { trainDataset, validationDataset, size: allTokens.length };
}
