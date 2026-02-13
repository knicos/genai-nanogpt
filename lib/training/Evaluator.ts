import { LazyIterator } from '@tensorflow/tfjs-data/dist/iterators/lazy_iterator';
import { Dataset } from '@tensorflow/tfjs-data';
import { tensor, Tensor, TensorContainer } from '@tensorflow/tfjs-core';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { calculateLoss } from './loss';
import { Conversation, ITokeniser } from '@base/main';
import { buildSFTExample } from './SFTDatasetBuilder';

export default class Evaluator {
    private iterator?: Promise<LazyIterator<TensorContainer>>;
    private xs?: Tensor;
    private ys?: Tensor;

    constructor(
        private model: Model<ModelForwardAttributes>,
        dataset: Dataset<TensorContainer> | Conversation[][],
        tokeniser?: ITokeniser
    ) {
        if (Array.isArray(dataset)) {
            if (!tokeniser) {
                throw new Error('Tokeniser is required when dataset is an array of conversations');
            }
            const example = dataset
                .map((data) => buildSFTExample(data, -100, tokeniser, model.config.blockSize))
                .filter((e) => e !== null);
            if (example.length === 0) {
                return;
            }
            this.xs = tensor(example.map((ex) => ex.xs));
            this.ys = tensor(example.map((ex) => ex.ys));
        } else {
            this.iterator = dataset.iterator();
        }
    }

    dispose() {
        if (this.xs) this.xs.dispose();
        if (this.ys) this.ys.dispose();
    }

    private async calculateBatchLoss(
        xs: Tensor,
        ys: Tensor,
        keepBatch: boolean,
        masked: boolean
    ): Promise<number | number[]> {
        const logits = this.model.forward({ training: false }, xs);
        const loss = calculateLoss(logits, ys, masked, keepBatch);
        logits.dispose();

        const lossValue = await loss!.array();
        const batchLoss = lossValue as number | number[];

        loss!.dispose();

        return batchLoss;
    }

    async evaluate(maxBatches = 100): Promise<number | number[]> {
        let totalLoss = 0;
        let batchCount = 0;

        if (this.iterator) {
            const iterator = await this.iterator;
            for (let i = 0; i < maxBatches; i++) {
                const result = await iterator.next();
                if (result.done) break;
                const batch = result.value;
                const { xs, ys } = batch as { xs: Tensor; ys: Tensor };

                //const logits = this.model.forward({ training: false }, xs);
                const loss = await this.calculateBatchLoss(xs, ys, false, false);
                xs.dispose();
                ys.dispose();

                totalLoss += loss as number;
                batchCount++;
            }

            return totalLoss / batchCount;
        } else if (this.xs && this.ys) {
            return this.calculateBatchLoss(this.xs, this.ys, true, true);
        }
        throw new Error('No data available for evaluation');
    }
}
