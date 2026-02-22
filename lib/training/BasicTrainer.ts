import type { ITokeniser } from '../tokeniser/type';
import Evaluator from './Evaluator';
import { dispose, keep, Scalar, scalar, Tensor, tidy, variableGrads, zeros } from '@tensorflow/tfjs-core';
import { Dataset } from '@tensorflow/tfjs-data';
import MemoryProfiler from '@base/utilities/profile';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { createTensorStatistics, TensorStatistics } from '../checks/weights';
import { NamedVariableMap } from '@tensorflow/tfjs-core/dist/tensor_types';
import { AdamWOptimizerConfig, TrainingLogEntry, TrainingMetrics, TrainingOptions, TrainingState } from './types';
import { calculateLoss } from './loss';
import { AdamWOptimizer } from './AdamW';

const DEFAULT_OPTIONS: TrainingOptions = {
    logInterval: 1,
    maxSteps: 1000,
    sftMode: 'full',
    batchSize: 32,
};

const DEFAULT_OPT_CONFIG: AdamWOptimizerConfig = {
    learningRate: 3e-4,
    beta1: 0.9,
    beta2: 0.99,
    epsilon: 1e-8,
    weightDecay: 0.01,
    warmupSteps: 100,
    decaySteps: 10000,
    minLearningRate: 1e-5,
    lossScaling: 1.0,
};

export default class BasicTrainer {
    public model: Model<ModelForwardAttributes>;
    public optimizer!: AdamWOptimizer;
    protected running = false;
    protected lastState?: TrainingState;
    protected _gradientCheckpointing = false;
    protected _mixedPrecision = false;
    protected maskedLoss = false;
    protected optimizerConfig: AdamWOptimizerConfig;
    protected metrics = new Set<TrainingMetrics>();

    constructor(
        model: Model<ModelForwardAttributes>,
        public tokenizer: ITokeniser,
        optConfig?: Partial<AdamWOptimizerConfig>,
        optimizer?: AdamWOptimizer
    ) {
        this.model = model;
        this.optimizerConfig = {
            ...DEFAULT_OPT_CONFIG,
            ...optConfig,
            lossScaling: model.lossScaling,
        };
        const adam = optimizer ? optimizer : new AdamWOptimizer(this.optimizerConfig);
        if (optimizer) {
            optimizer.updateConfig(this.optimizerConfig);
        }
        this.optimizer = adam;
    }

    setGradientCheckpointing(enabled: boolean): void {
        this._gradientCheckpointing = enabled;
    }

    setMixedPrecision(enabled: boolean): void {
        this._mixedPrecision = enabled;
    }

    setLearningRate(learningRate: number): void {
        this.optimizerConfig.learningRate = learningRate;
        this.updateOptimizer();
    }

    setMetrics(metrics: TrainingMetrics[]): void {
        this.metrics = new Set(metrics);
    }

    reset() {
        this.lastState = undefined;
        this.running = false;
    }

    stop() {
        this.running = false;
    }

    get isRunning(): boolean {
        return this.running;
    }

    getOptimizer(): AdamWOptimizer {
        return this.optimizer;
    }

    updateOptimizer(config?: Partial<AdamWOptimizerConfig>): void {
        if (config) {
            this.optimizerConfig = { ...this.optimizerConfig, ...config };
        }
        this.optimizer.updateConfig(this.optimizerConfig);
    }

    // A single forward pass, backward pass, and optimizer step
    protected trainStep(
        state: Partial<TrainingState>,
        batch: { xs: Tensor; ys: Tensor },
        dummy = false,
        keepGrads = false
    ): Scalar {
        return tidy(() => {
            this.model.getProfiler()?.startMemory();
            const { xs, ys } = batch;

            const f = () => {
                const logits = this.model.forward(
                    {
                        training: true,
                        checkpointing: this._gradientCheckpointing,
                        mixedPrecision: this._mixedPrecision,
                    },
                    xs
                );
                const loss = calculateLoss(logits, ys, this.maskedLoss);
                logits.dispose();
                const scaledLoss = loss.mul(scalar(this.optimizerConfig.lossScaling));
                loss.dispose();
                return scaledLoss as Scalar;
            };

            const { value: lossValue, grads } = variableGrads(f);

            if (!dummy) {
                // Apply gradients
                const scaling = this.optimizer.applyGradients(grads as NamedVariableMap);
                if (this.metrics.has('gradientNorm')) {
                    state.gradientNorm = scaling;
                    keep(scaling);
                } else {
                    state.gradientNorm = undefined;
                    scaling.dispose();
                }

                // Tell the model the weights were updated.
                const variableNames = Object.keys(grads);
                this.model.weightStore.touchVariables(variableNames);

                this.model.getProfiler()?.endMemory('Training');

                if (keepGrads) {
                    state.gradients = grads;
                    Object.values(grads).forEach((g) => keep(g));
                } else {
                    dispose(grads);
                }
            } else {
                this.model.getProfiler()?.endMemory('Training');
            }

            return lossValue.mul(scalar(1 / this.optimizerConfig.lossScaling)) as Scalar;
        });
    }

    private async dummyPass(): Promise<void> {
        // Send a dummy input to initialize the model
        const dummyBatch = zeros([1, this.model.config.blockSize], 'int32');
        const dummyTargets = zeros([1, this.model.config.blockSize], 'int32');

        try {
            const l = this.trainStep({}, { xs: dummyBatch, ys: dummyTargets }, true);
            await l.data(); // Ensure loss is computed
            l.dispose(); // Dispose loss to free memory
        } catch (error) {
            console.error('Error during dummy pass:', error);
        } finally {
            dummyBatch.dispose();
            dummyTargets.dispose(); // Dispose dummy targets to free memory
        }
    }

    dispose(): void {
        if (this.optimizer) {
            this.optimizer.dispose();
        }
    }

    private createEmptyState(): TrainingState {
        const state: TrainingState = {
            step: 0,
            lastLoss: 1e6,
            totalSteps: 0,
            losses: [],
            validationLosses: [],
            logStartTime: 0,
            trainingDuration: 0,
            ...(this.lastState || {}),
        };
        return state;
    }

    async stepDataset(
        dataset: Dataset<{ xs: Tensor; ys: Tensor }>,
        options: Partial<TrainingOptions>,
        validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>
    ): Promise<{ log: TrainingLogEntry }> {
        const { logInterval = 10 } = {
            ...DEFAULT_OPTIONS,
            ...options,
        };

        if (options.metrics) {
            this.setMetrics(options.metrics);
        }

        const startTime = Date.now();

        const state = this.createEmptyState();
        this.lastState = state;

        await this.dummyPass();
        // this.model.trainable = true;

        if (this.metrics.has('memoryUsage')) {
            if (!this.model.getProfiler()) {
                this.model.setProfiler(new MemoryProfiler());
            }
        }

        this.running = true;
        state.logStartTime = startTime;

        const evaluator = validationDataset ? new Evaluator(this.model, validationDataset) : undefined;
        const iterator = await dataset.iterator();

        try {
            while (this.running) {
                const result = await iterator.next();
                if (result.done) break;
                const batch = result.value;

                const lossScalar = this.trainStep(state, batch, false);
                batch.xs.dispose();
                batch.ys.dispose();

                state.step++;
                state.totalSteps++;

                if (state.step % logInterval === 0) {
                    await this.performLogging(lossScalar, batch.xs.shape[0], options, evaluator);
                } else {
                    if (state.gradientNorm) {
                        state.gradientNorm.dispose();
                        state.gradientNorm = undefined;
                    }
                }
                lossScalar.dispose();
            }
        } catch (error) {
            console.error('Training error:', error);
            dispose();
            throw error;
        }

        dispose();

        this.running = false;

        throw new Error('No log returned before training stopped.');
    }

    private async performLogging(
        lossScalar: Scalar,
        batchSize: number,
        options?: Partial<TrainingOptions>,
        evaluator?: Evaluator
    ): Promise<void> {
        const onStep = options?.onStep;
        const keepGrads = this.metrics.has('gradientStatistics');
        const lossValue = (await lossScalar.data())[0];
        const state = this.lastState!;
        state.lastLoss = lossValue;
        const logEndTime = Date.now();
        state.trainingDuration += logEndTime - state.logStartTime;

        const entry: TrainingLogEntry = {
            trainingMetrics: {
                loss: state.lastLoss,
                perplexity: this.metrics.has('perplexity') ? Math.exp(state.lastLoss) : undefined,
            },
            step: state.step,
            time: Date.now() - state.logStartTime,
            gradientNorm: state.gradientNorm ? (await state.gradientNorm.data())[1] : undefined,
            batchSize: batchSize,
            learningRate: this.metrics.has('learningRate') ? this.optimizer.lr : undefined,
            duration: state.trainingDuration,
            totalSamples: state.totalSteps * batchSize,
            samplesPerSecond: (state.totalSteps * batchSize) / (state.trainingDuration / 1000),
            memoryUsage: this.metrics.has('memoryUsage') ? this.model.getProfiler()?.getPeakMemory() || 0 : undefined,
        };
        if (this.metrics.has('tokensPerSecond')) {
            entry.tokensPerSecond = entry.samplesPerSecond * this.model.config.blockSize;
        }
        if (state.gradientNorm) {
            state.gradientNorm.dispose();
            state.gradientNorm = undefined;
        }

        this.model.trainingState = {
            steps: state.totalSteps,
            learningRate: this.optimizer.lr,
            batchSize: batchSize,
            loss: state.lastLoss,
        };

        if (keepGrads && state.gradients) {
            const gradMetrics = new Map<string, TensorStatistics>();
            for (const [name, grad] of Object.entries(state.gradients)) {
                gradMetrics.set(name, await createTensorStatistics(grad));
                grad.dispose();
            }
            entry.gradientMetrics = gradMetrics;
        }

        // Calculate validation loss if evaluator is provided
        if (evaluator) {
            try {
                const valLoss = await evaluator.evaluate(5);
                if (Array.isArray(valLoss)) {
                    entry.validationMetrics = { loss: valLoss[0] };
                } else {
                    state.validationLosses.push(valLoss);
                    entry.validationMetrics = {
                        loss: valLoss,
                        perplexity: this.metrics.has('perplexity') ? Math.exp(valLoss) : undefined,
                    };
                }
            } catch (error) {
                console.error('Validation error:', error);
            }
        }
        if (onStep) {
            await onStep(entry);
        }

        state.logStartTime = Date.now();
    }

    async trainOnDataset(
        dataset: Dataset<{ xs: Tensor; ys: Tensor }>,
        options: Partial<TrainingOptions>,
        validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>
    ): Promise<{ losses: number[]; validationLosses: number[] }> {
        const { logInterval = 10, maxSteps = Infinity } = {
            ...DEFAULT_OPTIONS,
            ...options,
        };

        if (options.metrics) {
            this.setMetrics(options.metrics);
        }

        const startTime = Date.now();

        const state = this.createEmptyState();
        this.lastState = state;

        await this.dummyPass();
        // this.model.trainable = true;

        if (options?.metrics?.includes('memoryUsage')) {
            if (!this.model.getProfiler()) {
                this.model.setProfiler(new MemoryProfiler());
            }
        }

        this.running = true;
        state.logStartTime = startTime;

        const evaluator = validationDataset ? new Evaluator(this.model, validationDataset) : undefined;
        const iterator = await dataset.iterator();

        try {
            while (this.running) {
                const result = await iterator.next();
                if (result.done) break;
                const batch = result.value;
                const isLogStep = state.step % logInterval === 0;
                const keepGrads = (options?.metrics?.includes('gradientStatistics') || false) && isLogStep;

                // Do the actual training step
                const lossScalar = this.trainStep(state, batch, false, keepGrads);
                batch.xs.dispose();
                batch.ys.dispose();

                state.step++;
                state.totalSteps++;

                if (isLogStep) {
                    await this.performLogging(lossScalar, batch.xs.shape[0], options, evaluator);
                } else {
                    if (state.gradientNorm) {
                        state.gradientNorm.dispose();
                        state.gradientNorm = undefined;
                    }
                }
                lossScalar.dispose();

                if (state.step >= maxSteps) {
                    this.stop();
                }
            }
        } catch (error) {
            console.error('Training error:', error);
            dispose();
            throw error;
        }

        dispose();

        this.running = false;

        return { losses: state.losses, validationLosses: state.validationLosses };
    }
}
