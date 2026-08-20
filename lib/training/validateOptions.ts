import { TrainingOptions } from './types';

export default function validateOptions(
    options: TrainingOptions,
    training: boolean,
    oldOptions?: TrainingOptions
): TrainingOptions {
    // Check which options have changed and only update those
    // This allows us to change options like learning rate or metrics on the fly without resetting the entire trainer
    // Throw if some options are changed during training such as batchSize.

    const changedOptions = new Set(
        Object.keys(options).filter(
            (key) => options[key as keyof TrainingOptions] !== oldOptions?.[key as keyof TrainingOptions]
        )
    );

    if (training) {
        if (changedOptions.has('batchSize')) {
            throw new Error('Cannot change batch size during training');
        }
        if (changedOptions.has('sftMode')) {
            throw new Error('Cannot change SFT mode during training');
        }
        if (changedOptions.has('loraConfig')) {
            throw new Error('Cannot change LoRA configuration during training');
        }
        if (changedOptions.has('validationSplit')) {
            throw new Error('Cannot change validation split during training');
        }
        if (changedOptions.has('trainableWeights')) {
            throw new Error('Cannot change trainable weights during training');
        }
        if (changedOptions.has('mixedPrecision')) {
            throw new Error('Cannot change mixed precision setting during training');
        }
        if (changedOptions.has('gradientCheckpointing')) {
            throw new Error('Cannot change gradient checkpointing setting during training');
        }
    }

    return {
        ...oldOptions,
        ...options,
    };
}
