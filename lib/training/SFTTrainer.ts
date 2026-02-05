import Model, { ModelForwardAttributes } from '@base/models/model';
import BasicTrainer from './BasicTrainer';
import { ITokeniser } from '@base/tokeniser/type';
import { AdamConfig } from './types';
import AdamExt from './AdamExt';
import { SFTDatasetBuilder } from './SFTDatasetBuilder';

export default class SFTTrainer extends BasicTrainer {
    public datasetBuilder: SFTDatasetBuilder;

    constructor(
        model: Model<ModelForwardAttributes>,
        public tokenizer: ITokeniser,
        learningRate: number = 3e-4
    ) {
        super(model, tokenizer, learningRate);
        this.datasetBuilder = new SFTDatasetBuilder(tokenizer, model.config.blockSize);
        this.maskedLoss = true;
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
                warmupSteps: 0,
                decaySteps: 0,
                minLearningRate: this.learningRate * config.learningRateFactor,
                weightDecay: 0.1,
                lossScaling: this.lossScaling,
            }
        );
        this.optimizer = adam;
    }
}
