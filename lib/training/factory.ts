import type { ModelForwardAttributes } from '@base/models/model';
import type { TrainingOptions } from '@base/training/types';
import BasicTrainer from '@base/training/BasicTrainer';
import SFTTrainer from '@base/training/SFTTrainer';
import PreTrainer from '@base/training/PreTrainer';
import Model from '@base/models/model';
import { ITokeniser } from '@base/tokeniser/type';
import { AdamWOptimizer } from './AdamW';

export default function createTrainer(
    model: Model<ModelForwardAttributes>,
    tokenizer: ITokeniser,
    options?: TrainingOptions,
    optimizer?: AdamWOptimizer
): BasicTrainer {
    if (options?.method.type === 'supervised') {
        return new SFTTrainer(model, tokenizer, options, optimizer);
    } else {
        return new PreTrainer(model, tokenizer, options, optimizer);
    }
}
