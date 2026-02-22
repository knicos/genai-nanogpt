import Model, { ModelForwardAttributes } from '@base/models/model';
import BasicTrainer from './BasicTrainer';
import { ITokeniser } from '@base/tokeniser/type';
import { DatasetBuilder } from './DatasetBuilder';
import { AdamWOptimizer } from './AdamW';
import { AdamWOptimizerConfig } from './types';

const DEFAULT_OPT_CONFIG: Partial<AdamWOptimizerConfig> = {
    decaySteps: 60000,
    warmupSteps: 1000,
    minLearningRate: 3e-5,
    weightDecay: 0.1,
    learningRate: 3e-4,
    // clipNorm: 1.0,
};

export default class PreTrainer extends BasicTrainer {
    public datasetBuilder: DatasetBuilder;

    constructor(
        model: Model<ModelForwardAttributes>,
        public tokenizer: ITokeniser,
        optConfig?: Partial<AdamWOptimizerConfig>,
        optimizer?: AdamWOptimizer
    ) {
        super(model, tokenizer, { ...DEFAULT_OPT_CONFIG, ...optConfig }, optimizer);

        this.optimizerConfig.minLearningRate = this.optimizerConfig.learningRate / 10;

        this.updateOptimizer();
        this.datasetBuilder = new DatasetBuilder(tokenizer, model.config.blockSize);
    }
}
