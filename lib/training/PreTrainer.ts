import Model, { ModelForwardAttributes } from '@base/models/model';
import BasicTrainer from './BasicTrainer';
import { ITokeniser } from '@base/tokeniser/type';
import { DatasetBuilder } from './DatasetBuilder';
import { AdamConfig } from './types';
import AdamExt from './AdamExt';

export default class PreTrainer extends BasicTrainer {
    public datasetBuilder: DatasetBuilder;

    constructor(
        model: Model<ModelForwardAttributes>,
        public tokenizer: ITokeniser,
        learningRate: number = 3e-4
    ) {
        super(model, tokenizer, learningRate);
        this.resetOptimizer();
        this.datasetBuilder = new DatasetBuilder(tokenizer, model.config.blockSize);
    }

    override resetOptimizer(
        config: AdamConfig = { learningRateFactor: 1, beta1: 0.9, beta2: 0.99, epsilon: 1e-8 }
    ): void {
        if (this.optimizer) this.optimizer.dispose();
        const adam = new AdamExt(
            config.learningRateFactor * this.learningRate,
            config.beta1,
            config.beta2,
            config.epsilon,
            {
                warmupSteps: 100,
                decaySteps: 20000,
                minLearningRate: 1e-4,
                weightDecay: 0,
                lossScaling: this.lossScaling,
            }
        );
        this.optimizer = adam;
    }
}
