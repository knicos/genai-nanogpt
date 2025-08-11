import NanoGPT from '../NanoGPTModel';
import type TF from '@tensorflow/tfjs';
import { LazyIterator } from '@tensorflow/tfjs-data/dist/iterators/lazy_iterator';

export default class Evaluator {
    private iterator: Promise<LazyIterator<TF.TensorContainer>>;

    constructor(private model: NanoGPT, dataset: TF.data.Dataset<TF.TensorContainer>) {
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
            const { xs, ys } = batch as { xs: TF.Tensor; ys: TF.Tensor };

            const { loss, logits } = this.model.forward(xs, ys, false, false);
            logits.dispose();
            xs.dispose();
            ys.dispose();

            const lossValue = loss!.arraySync();
            const batchLoss = lossValue as number;

            loss!.dispose();

            totalLoss += batchLoss;
            batchCount++;
        }

        return totalLoss / batchCount;
    }
}
