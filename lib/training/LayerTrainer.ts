import { ITokeniser } from '../tokeniser/type';
import { generateText } from '../utilities/generate';
import NanoGPT, { TrainingLogEntry } from '../NanoGPTModel';
import type TF from '@tensorflow/tfjs';
import GPTTrainer, { TrainingOptions, TrainingState } from './Trainer';
import { schedule, LWSchedule } from './lwSchedule';

interface LayerTrainingState extends TrainingState {
    epoch: number;
    pass: number;
    layerStep: number;
    step: number;
    stepSinceLayerChange: number;
    lastLoss: number;
    epochLoss: number;
    totalSteps: number;
    losses: number[];
    validationLosses: number[];
}

interface LayerTrainingLogEntry extends TrainingLogEntry {
    pass: number;
    layer: number;
}

interface LayerTrainingOptions extends TrainingOptions {
    stepsPerLayer: number;
    maxPasses: number;
    onLayerChange?: (layer: number, pass: number, valLoss?: number) => Promise<void> | void;
    onPassComplete?: (pass: number) => Promise<void> | void;
}

const DEFAULT_OPTIONS: LayerTrainingOptions = {
    epochs: 1,
    stepsPerEpoch: 1000000,
    desiredLoss: 0.01,
    logInterval: 1,
    stepsPerLayer: 400,
    maxPasses: 3,
};

// Enhanced training utilities with Dataset API and memory leak fixes
export default class LayerTrainer extends GPTTrainer {
    private trainingPattern: LWSchedule[] = [];
    private startPass: number = 0;
    private startLayer: number = 0;

    constructor(tf: typeof TF, model: NanoGPT, tokenizer: ITokeniser, learningRate: number = 3e-4) {
        super(tf, model, tokenizer, learningRate);

        this.trainingPattern = schedule[model.config.nLayer - 1] || [];

        if (model.log.length > 0) {
            const lastEntry = model.log[model.log.length - 1] as LayerTrainingLogEntry;
            if (lastEntry.pass !== undefined && lastEntry.layer !== undefined) {
                // Resume from last training state
                this.startPass = lastEntry.pass;
                this.startLayer = lastEntry.layer;
                // TODO: How far through the layer? Move to next layer if needed

                console.log(`Resuming training from pass ${this.startPass}, layer ${this.startLayer}`);
            }
        }
    }

    private applyTrainingPattern(pass: number) {
        const ix = pass < this.trainingPattern.length ? pass : this.trainingPattern.length - 1;
        const pattern = this.trainingPattern[ix];
        this.model.setSkipMask(pattern.skip);
        this.model.setTrainableMask(pattern.trainable);
        this.resetOptimizer(pattern.adam);
        console.log('Applied training pattern:', ix, pattern);
    }

    // Train for multiple epochs using Dataset API - FIXED memory leaks
    async trainOnDataset(
        dataset: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>,
        options: Partial<LayerTrainingOptions>,
        validationDataset?: TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>
    ): Promise<{ losses: number[]; validationLosses: number[] }> {
        const {
            epochs,
            stepsPerEpoch,
            desiredLoss,
            logInterval,
            stepsPerLayer,
            onLayerChange,
            onPassComplete,
            onStep,
            onEpoch,
            prompt,
        } = {
            ...DEFAULT_OPTIONS,
            ...options,
        };

        const state: LayerTrainingState = {
            epoch: 0,
            pass: 0,
            layerStep: 0,
            step: 0,
            stepSinceLayerChange: 0,
            lastLoss: 1e6,
            epochLoss: 0,
            totalSteps: 0,
            losses: [],
            validationLosses: [],
        };

        this.dummyPass();

        const startTime = Date.now();

        for (state.epoch = 0; state.epoch < epochs; state.epoch++) {
            state.step = 0;
            state.epochLoss = 0;
            state.pass = this.startPass;
            state.layerStep = this.startLayer + this.startPass * this.model.config.nLayer;
            state.stepSinceLayerChange = 0;
            this.startPass = 0;
            this.startLayer = 0;

            const iterator = await dataset.iterator();

            this.applyTrainingPattern(state.layerStep % this.trainingPattern.length);

            // Training loop with try-catch for better error handling
            try {
                while (true) {
                    if (stepsPerEpoch && state.step >= stepsPerEpoch) break;
                    if (state.lastLoss < desiredLoss) break;

                    const result = await iterator.next();
                    if (result.done) break;
                    const batch = result.value;

                    const prom = this.trainBatch(state, batch);
                    state.stepSinceLayerChange++;

                    const entry: LayerTrainingLogEntry = {
                        epoch: state.epoch,
                        loss: state.lastLoss,
                        step: state.step,
                        time: Date.now() - startTime,
                        batchSize: batch.xs.shape[0],
                        pass: state.pass,
                        layer: state.layerStep % this.model.config.nLayer,
                    };
                    this.model.log.push(entry);

                    if (state.step % logInterval === 0) {
                        await prom;
                        if (onStep) {
                            if (prompt) {
                                const text = await generateText(this.tokenizer, this.model, prompt, 100, {
                                    temperature: 0.8,
                                    topK: 10,
                                });
                                entry.example = text;
                            }
                            await onStep(entry);
                        }
                    }

                    if (state.stepSinceLayerChange >= stepsPerLayer) {
                        let valLoss: number | undefined;
                        if (validationDataset) {
                            valLoss = await this.evaluateOnDataset(validationDataset, 5);
                            state.validationLosses.push(valLoss);
                            entry.valLoss = valLoss;
                        }

                        state.layerStep++;
                        const passComplete = state.layerStep % this.model.config.nLayer === 0;
                        if (!passComplete) {
                            if (onLayerChange) {
                                await onLayerChange(state.layerStep, state.pass, valLoss);
                            }
                        } else {
                            if (onLayerChange) {
                                await onLayerChange(state.layerStep, state.pass, valLoss);
                            }
                            if (onPassComplete) {
                                await onPassComplete(state.pass);
                            }
                            state.pass++;
                        }
                        state.stepSinceLayerChange = 0;
                        this.applyTrainingPattern(state.layerStep % this.trainingPattern.length);
                    }
                }
            } catch (error) {
                console.error('Training error:', error);
                this.tf.dispose();
                throw error;
            }

            const avgLoss = state.epochLoss / state.step;

            // Validation
            if (validationDataset) {
                try {
                    const valLoss = await this.evaluateOnDataset(validationDataset, 5);
                    state.validationLosses.push(valLoss);

                    if (onEpoch) {
                        await onEpoch(state.epoch, avgLoss, valLoss);
                    }
                } catch (error) {
                    console.error('Validation error:', error);
                }
            } else {
                if (onEpoch) {
                    onEpoch(state.epoch, avgLoss);
                }
            }

            this.tf.dispose();

            if (state.lastLoss < desiredLoss) {
                break;
            }
        }

        return { losses: state.losses, validationLosses: state.validationLosses };
    }
}
