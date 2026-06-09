import Model, { ModelForwardAttributes } from '@base/models/model';
import BasicTrainer from './BasicTrainer';
import { ITokeniser } from '@base/tokeniser/type';
import { AdamWOptimizer } from './AdamW';
import { AdamWOptimizerConfig } from './types';
import { DatasetBuilder } from './DatasetBuilder';

const DEFAULT_OPT_CONFIG: Partial<AdamWOptimizerConfig> = {
    decayEpochs: 100,
    epochSteps: 10000,
    warmupSteps: 100,
    minLearningRate: 1e-5,
    weightDecay: 0.1,
    beta2: 0.95,
    learningRate: 3e-4,
    // clipNorm: 1.0,
};

export default class SFTTrainer extends BasicTrainer {
    public datasetBuilder: DatasetBuilder;
    public loraName?: string;

    constructor(
        model: Model<ModelForwardAttributes>,
        public tokenizer: ITokeniser,
        optConfig?: Partial<AdamWOptimizerConfig>,
        optimizer?: AdamWOptimizer
    ) {
        super(model, tokenizer, { ...DEFAULT_OPT_CONFIG, ...optConfig }, optimizer);

        this.optimizerConfig.minLearningRate = optConfig?.minLearningRate ?? this.optimizerConfig.learningRate / 20;
        this.updateOptimizer();
        this.datasetBuilder = new DatasetBuilder(tokenizer, model.config.blockSize);
        this.maskedLoss = true;
    }
}
