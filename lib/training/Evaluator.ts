import { LazyIterator } from '@tensorflow/tfjs-data/dist/iterators/lazy_iterator';
import { Dataset } from '@tensorflow/tfjs-data';
import { Tensor, TensorContainer } from '@tensorflow/tfjs-core';
import Model, { ModelForwardAttributes } from '@base/models/model';

export default class Evaluator {
    private iterator: Promise<LazyIterator<TensorContainer>>;

    constructor(private model: Model<ModelForwardAttributes>, dataset: Dataset<TensorContainer>) {
        this.iterator = dataset.iterator();
    }

    async evaluate(maxBatches: number = 100): Promise<number> {
        let totalLoss = 0;
        let batchCount = 0;

        const iterator = await this.iterator;
        for (let i = 0; i < maxBatches; i++) {
            const result = await iterator.next();
            if (result.done) break;
            const batch = result.value;
            const { xs, ys } = batch as { xs: Tensor; ys: Tensor };

            const [logits, loss] = this.model.forward({ training: false }, xs, ys);
            logits.dispose();
            xs.dispose();
            ys.dispose();

            const lossValue = await loss!.array();
            const batchLoss = lossValue as number;

            loss!.dispose();

            totalLoss += batchLoss;
            batchCount++;
        }

        return totalLoss / batchCount;
    }
}
