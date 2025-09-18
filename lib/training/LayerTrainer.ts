import type { ITokeniser } from '../tokeniser/type';
import { generateText } from '../utilities/generate';
import NanoGPT, { TrainingLogEntry } from '../NanoGPTModel';
import GPTTrainer, { TrainingOptions, TrainingState } from './Trainer';
import { schedule, LWSchedule } from './lwSchedule';
import Evaluator from './Evaluator';
import { dispose, Tensor } from '@tensorflow/tfjs-core';
import { Dataset } from '@tensorflow/tfjs-data';

interface LayerTrainingState extends TrainingState {
    pass: number;
    layerStep: number;
    step: number;
    stepSinceLayerChange: number;
    lastLoss: number;
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
    desiredLoss: 0.01,
    logInterval: 1,
    stepsPerLayer: 400,
    maxPasses: 3,
    maxSteps: 1000,
};

// Enhanced training utilities with Dataset API and memory leak fixes
export default class LayerTrainer extends GPTTrainer {
    private trainingPattern: LWSchedule[] = [];
    private startPass: number = 0;
    private startLayer: number = 0;

    constructor(model: NanoGPT, tokenizer: ITokeniser, learningRate: number = 3e-4) {
        super(model, tokenizer, learningRate);

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
        dataset: Dataset<{ xs: Tensor; ys: Tensor }>,
        options: Partial<LayerTrainingOptions>,
        validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>
    ): Promise<{ losses: number[]; validationLosses: number[] }> {
        const { desiredLoss, logInterval, stepsPerLayer, onLayerChange, onPassComplete, onStep, prompt } = {
            ...DEFAULT_OPTIONS,
            ...options,
        };

        const state: LayerTrainingState = {
            pass: 0,
            layerStep: 0,
            step: 0,
            stepSinceLayerChange: 0,
            lastLoss: 1e6,
            totalSteps: 0,
            losses: [],
            validationLosses: [],
        };

        this.dummyPass();

        const startTime = Date.now();
        this.startPass = 0;
        this.startLayer = 0;

        const evaluator = validationDataset ? new Evaluator(this.model, validationDataset) : undefined;

        const iterator = await dataset.iterator();

        this.applyTrainingPattern(state.layerStep % this.trainingPattern.length);

        // Training loop with try-catch for better error handling
        try {
            while (true) {
                if (state.lastLoss < desiredLoss) break;

                const result = await iterator.next();
                if (result.done) break;
                const batch = result.value;

                const prom = this.trainBatch(state, batch);
                state.stepSinceLayerChange++;

                const entry: LayerTrainingLogEntry = {
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
                    // Validation
                    if (evaluator) {
                        try {
                            const valLoss = await evaluator.evaluate(5);
                            state.validationLosses.push(valLoss);
                            entry.valLoss = valLoss;
                        } catch (error) {
                            console.error('Validation error:', error);
                        }
                    }
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
                    state.layerStep++;
                    const passComplete = state.layerStep % this.model.config.nLayer === 0;
                    if (!passComplete) {
                        if (onLayerChange) {
                            await onLayerChange(state.layerStep, state.pass);
                        }
                    } else {
                        if (onLayerChange) {
                            await onLayerChange(state.layerStep, state.pass);
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
            dispose();
            throw error;
        }

        dispose();

        return { losses: state.losses, validationLosses: state.validationLosses };
    }
}
