import type { TrainingOptions } from './types';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { v4 as uuidv4 } from 'uuid';

export default function configureModel(model: Model<ModelForwardAttributes>, options?: TrainingOptions) {
    const mode = options?.method.supervised || 'full';
    const type = options?.method.type || 'pretraining';

    if (type === 'pretraining') {
        if (model.hasLoRA()) {
            model.detachLoRA();
        }
        model.weightStore.setTrainable(['*']);

        if (options) {
            model.metaData.pretrainingSettings = options;
        }
    }

    if (type === 'supervised') {
        if (mode === 'lora') {
            if (options?.loraName) {
                if (!model.hasLoRA(options.loraName)) {
                    if (options.loraConfig) {
                        model.createLoRA(options.loraName, options.loraConfig);
                        model.attachLoRA(options.loraName);
                    } else {
                        throw new Error(
                            `LoRA configuration must be provided to create LoRA with name ${options.loraName}`
                        );
                    }
                } else {
                    model.attachLoRA(options.loraName);
                    if (options.loraConfig) {
                        const existingLoRA = model.lora!;
                        if (
                            existingLoRA.alpha !== options.loraConfig.alpha ||
                            existingLoRA.rank !== options.loraConfig.rank
                        ) {
                            // Reset to a new LoRA
                            model.detachLoRA();
                            model.deleteLoRA(options.loraName);
                            model.createLoRA(options.loraName, options.loraConfig);
                            model.attachLoRA(options.loraName);
                            console.warn('Resetting LoRA with new configuration.');
                        }
                    }
                }
            } else if (options?.loraConfig) {
                if (model.hasLoRA()) {
                    const existingLoRA = model.lora!;
                    if (
                        existingLoRA.alpha !== options.loraConfig.alpha ||
                        existingLoRA.rank !== options.loraConfig.rank
                    ) {
                        // Reset to a new LoRA
                        model.detachLoRA();
                        const loraName = options.loraName || uuidv4();
                        model.createLoRA(loraName, options.loraConfig);
                        model.attachLoRA(loraName);
                    }
                } else {
                    const loraName = options.loraName || uuidv4();
                    model.createLoRA(loraName, options.loraConfig);
                    model.attachLoRA(loraName);
                }
            } else if (model.hasLoRA()) {
                // Keep existing LoRA
            } else {
                throw new Error('LoRA configuration must be provided for lora SFT mode');
            }
        } else {
            if (model.hasLoRA()) {
                model.detachLoRA();
            }
        }

        if (mode === 'last-layer') {
            model.weightStore.setTrainable([`block_${model.config.nLayer - 1}_*`, 'token_embedding']);
        } else if (mode === 'full') {
            model.weightStore.setTrainable(['*']);
        }
    }

    if (options?.trainableWeights) {
        model.weightStore.setTrainable(options.trainableWeights);
    }
}
