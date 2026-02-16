import Model, { ModelForwardAttributes } from '@base/models/model';
import BasicTrainer from './BasicTrainer';
import { ITokeniser } from '@base/tokeniser/type';
import { SFTDatasetBuilder } from './SFTDatasetBuilder';
import { AdamWOptimizerConfig } from './AdamW';

const DEFAULT_OPT_CONFIG: Partial<AdamWOptimizerConfig> = {
    decaySteps: 10000,
    warmupSteps: 100,
    minLearningRate: 1e-5,
    weightDecay: 0.1,
    beta2: 0.95,
    learningRate: 3e-4,
    // clipNorm: 1.0,
};

export default class SFTTrainer extends BasicTrainer {
    public datasetBuilder: SFTDatasetBuilder;

    constructor(
        model: Model<ModelForwardAttributes>,
        public tokenizer: ITokeniser,
        optConfig?: Partial<AdamWOptimizerConfig>
    ) {
        super(model, tokenizer, { ...DEFAULT_OPT_CONFIG, ...optConfig });

        // this.resetOptimizer();

        this.datasetBuilder = new SFTDatasetBuilder(tokenizer, model.config.blockSize);
        this.maskedLoss = true;
    }
}
