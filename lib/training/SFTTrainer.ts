import Model, { ModelForwardAttributes } from '@base/models/model';
import BasicTrainer from './BasicTrainer';
import { ITokeniser } from '@base/tokeniser/type';
import { SFTDatasetBuilder } from './SFTDatasetBuilder';

export default class SFTTrainer extends BasicTrainer {
    public datasetBuilder: SFTDatasetBuilder;

    constructor(
        model: Model<ModelForwardAttributes>,
        public tokenizer: ITokeniser,
        learningRate = 3e-4
    ) {
        super(model, tokenizer);

        this.optimizerConfig.decaySteps = 10000;
        this.optimizerConfig.warmupSteps = 100;
        this.optimizerConfig.minLearningRate = 1e-5;
        this.optimizerConfig.weightDecay = 0.1;
        this.optimizerConfig.beta2 = 0.95;
        this.optimizerConfig.learningRate = learningRate;

        this.resetOptimizer();

        this.datasetBuilder = new SFTDatasetBuilder(tokenizer, model.config.blockSize);
        this.maskedLoss = true;
    }
}
