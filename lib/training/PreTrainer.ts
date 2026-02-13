import Model, { ModelForwardAttributes } from '@base/models/model';
import BasicTrainer from './BasicTrainer';
import { ITokeniser } from '@base/tokeniser/type';
import { DatasetBuilder } from './DatasetBuilder';

export default class PreTrainer extends BasicTrainer {
    public datasetBuilder: DatasetBuilder;

    constructor(
        model: Model<ModelForwardAttributes>,
        public tokenizer: ITokeniser,
        learningRate = 3e-4
    ) {
        super(model, tokenizer);

        this.optimizerConfig.decaySteps = 200000;
        this.optimizerConfig.warmupSteps = 100;
        this.optimizerConfig.minLearningRate = 1e-4;
        this.optimizerConfig.weightDecay = 0.1;
        this.optimizerConfig.learningRate = learningRate;

        this.resetOptimizer();
        this.datasetBuilder = new DatasetBuilder(tokenizer, model.config.blockSize);
    }
}
