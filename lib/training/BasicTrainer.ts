import type { ITokeniser } from '../tokeniser/type';
import Evaluator from './Evaluator';
import { dispose, keep, Scalar, scalar, Tensor, tidy, variableGrads, zeros } from '@tensorflow/tfjs-core';
import { Dataset } from '@tensorflow/tfjs-data';
import MemoryProfiler from '@base/utilities/profile';
import Model, { ModelForwardAttributes } from '@base/models/model';
import { createTensorStatistics, TensorStatistics } from '../checks/weights';
import { NamedVariableMap } from '@tensorflow/tfjs-core/dist/tensor_types';
import { AdamWOptimizerConfig, TrainingLogEntry, TrainingMetrics, TrainingOptions, TrainingState } from './types';
import { calculateAccuracy, calculateLoss } from './loss';
import { AdamWOptimizer } from './AdamW';
import configureModel from './configure';

const DEFAULT_OPTIONS: TrainingOptions = {
    logInterval: 200,
    maxEpochs: 100,
    method: { type: 'pretraining' },
    batchSize: 32,
};

const DEFAULT_OPT_CONFIG: AdamWOptimizerConfig = {
    learningRate: 3e-4,
    beta1: 0.9,
    beta2: 0.99,
    epsilon: 1e-8,
    weightDecay: 0.01,
    warmupSteps: 100,
    decayEpochs: 100,
    epochSteps: 10000,
    minLearningRate: 1e-5,
    lossScaling: 1.0,
};

export default class BasicTrainer {
    public model: Model<ModelForwardAttributes>;
    public optimizer!: AdamWOptimizer;
    public log: TrainingLogEntry[] = [];
    protected running = false;
    protected lastState?: TrainingState;
    protected _gradientCheckpointing = false;
    protected _mixedPrecision = false;
    protected maskedLoss = false;
    protected optimizerConfig: AdamWOptimizerConfig;
    protected metrics = new Set<TrainingMetrics>();
    protected _labelSmoothing = 0.0;
    protected _layerDrop = 0.0;
    protected _dropout = 0.0;
    protected _tokensProcessed = 0;

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
            lossScaling: optConfig?.lossScaling ?? model.lossScaling,
        };
        const adam = optimizer ? optimizer : new AdamWOptimizer(this.optimizerConfig);
        if (optimizer) {
            optimizer.updateConfig(this.optimizerConfig);
        }
        this.optimizer = adam;
    }

    setLossMasking() {
        this.maskedLoss = true;
    }

    setGradientCheckpointing(enabled: boolean): void {
        this._gradientCheckpointing = enabled;
    }

    setMixedPrecision(enabled: boolean): void {
        this._mixedPrecision = enabled;
    }

    setLabelSmoothing(smoothing: number): void {
        this._labelSmoothing = smoothing;
    }

    setDropout(dropout: number): void {
        this._dropout = dropout;
    }

    setLayerDrop(layerDrop: number): void {
        this._layerDrop = layerDrop;
    }

    setLearningRate(learningRate: number): void {
        this.optimizerConfig.learningRate = learningRate;
        this.updateOptimizer();
    }

    setMetrics(metrics: TrainingMetrics[]): void {
        this.metrics = new Set(metrics);
    }

    configure(options: TrainingOptions) {
        this.setGradientCheckpointing(options?.gradientCheckpointing || false);
        this.setMixedPrecision(options?.mixedPrecision || false);
        this.setLabelSmoothing(options?.labelSmoothing || 0.0);
        this.setDropout(options?.dropout || 0);
        this.setLayerDrop(options?.layerDrop || 0);
        if (!this.lastState) {
            this.setLearningRate(options?.learningRate || 1e-3);
        }
        configureModel(this.model, options);
    }

    reset() {
        this.lastState = undefined;
        this.running = false;
        this.log = [];
    }

    stop() {
        this.running = false;
    }

    get isRunning(): boolean {
        return this.running;
    }

    get tokensProcessed(): number {
        return this._tokensProcessed;
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

    resumeFromLog(log: TrainingLogEntry): void {
        if (!this.lastState || this.lastState.step === 0) {
            this.lastState = {
                losses: [],
                validationLosses: [],
                logStartTime: 0,
                step: log.step,
                lastLoss: log.trainingMetrics.loss,
                totalSteps: log.step,
                trainingDuration: log.duration,
            };
        }
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

            // const randomRoPEOffset = Math.floor(Math.random() * this.model.config.blockSize * 2);

            const f = () => {
                const logits = this.model.forward(
                    {
                        training: true,
                        checkpointing: this._gradientCheckpointing,
                        mixedPrecision: this._mixedPrecision,
                        dropout: this._dropout,
                        layerDrop: this._layerDrop,
                        ropePositionOffset: 0,
                    },
                    xs
                );
                const loss = calculateLoss(logits, ys, this.maskedLoss, false, this._labelSmoothing);

                if (this.metrics.has('accuracy')) {
                    state.accuracy = calculateAccuracy(logits, ys);
                    keep(state.accuracy);
                }

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

    private async performLogging(
        lossScalar: Scalar,
        batchSize: number,
        evaluator?: Evaluator,
        onStep?: (log: TrainingLogEntry) => void
    ): Promise<void> {
        const keepGrads = this.metrics.has('gradientStatistics');
        const state = this.lastState!;

        // Collect async tensor reads up-front. Use null placeholders for unused reads.
        const promises: (Promise<Float32Array> | null)[] = [];
        promises.push(lossScalar.data<'float32'>());
        promises.push(state.accuracy ? state.accuracy.data<'float32'>() : null);
        promises.push(state.gradientNorm ? state.gradientNorm.data<'float32'>() : null);

        const results = await Promise.all(promises);

        const lossValue = results[0]?.[0] ?? 0;
        const accuracyValue = results[1] ? results[1][0] : undefined;
        const gradientNormValue = results[2] ? results[2][1] : undefined;

        state.lastLoss = lossValue;
        const logEndTime = Date.now();
        state.trainingDuration += logEndTime - state.logStartTime;
        const tokensProcessed = state.totalSteps * batchSize * this.model.config.blockSize;
        this._tokensProcessed = tokensProcessed;

        const entry: TrainingLogEntry = {
            trainingMetrics: {
                loss: state.lastLoss,
                perplexity: this.metrics.has('perplexity') ? Math.exp(state.lastLoss) : undefined,
                accuracy: accuracyValue,
            },
            step: state.step,
            time: Date.now() - state.logStartTime,
            gradientNorm: gradientNormValue,
            batchSize: batchSize,
            learningRate: this.metrics.has('learningRate') ? this.optimizer.lr : undefined,
            duration: state.trainingDuration,
            totalTokens: tokensProcessed,
            tokensPerSecond: tokensProcessed / (state.trainingDuration / 1000),
            memoryUsage: this.metrics.has('memoryUsage') ? this.model.getProfiler()?.getPeakMemory() || 0 : undefined,
        };

        if (state.gradientNorm) {
            state.gradientNorm.dispose();
            state.gradientNorm = undefined;
        }
        if (state.accuracy) {
            state.accuracy.dispose();
            state.accuracy = undefined;
        }

        this.model.trainingState = {
            steps: state.totalSteps,
            learningRate: this.optimizer.lr,
            batchSize: batchSize,
            loss: state.lastLoss,
            tokensProcessed,
            duration: state.trainingDuration,
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
                    entry.validationMetrics = { loss: valLoss[0].loss, accuracy: valLoss[0].accuracy };
                } else {
                    state.validationLosses.push(valLoss.loss);
                    entry.validationMetrics = {
                        accuracy: valLoss.accuracy,
                        loss: valLoss.loss,
                        perplexity: this.metrics.has('perplexity') ? Math.exp(valLoss.loss) : undefined,
                    };
                }
            } catch (error) {
                console.error('Validation error:', error);
            }
        }

        this.log.push(entry);

        if (onStep) {
            onStep(entry);
        }

        state.logStartTime = Date.now();
    }

    async trainOnDataset(
        dataset: Dataset<{ xs: Tensor; ys: Tensor }>,
        options: Partial<TrainingOptions>,
        validationDataset?: Dataset<{ xs: Tensor; ys: Tensor }>,
        onStep?: (log: TrainingLogEntry) => void
    ): Promise<{ losses: number[]; validationLosses: number[] }> {
        const { logInterval = 40, maxEpochs = Infinity } = {
            ...DEFAULT_OPTIONS,
            ...options,
        };

        // Ensure trainer state is resumed
        if (this.log.length > 0) {
            this.resumeFromLog(this.log[this.log.length - 1]);
        }

        const maxSteps = maxEpochs * (options?.epochSteps || 1000);

        if (options.metrics) {
            this.setMetrics(options.metrics);
        }

        const state = this.createEmptyState();
        this.lastState = state;

        if (state.step >= maxSteps) {
            return { losses: state.losses, validationLosses: state.validationLosses };
        }

        await this.dummyPass();
        // this.model.trainable = true;

        if (options?.metrics?.includes('memoryUsage')) {
            if (!this.model.getProfiler()) {
                this.model.setProfiler(new MemoryProfiler());
            }
        }

        const startTime = Date.now();
        this.running = true;
        state.logStartTime = startTime;
        let lastLog = startTime;

        const evaluator = validationDataset ? new Evaluator(this.model, validationDataset, this.maskedLoss) : undefined;
        const iterator = await dataset.iterator();

        let resultPromise = iterator.next();

        try {
            while (this.running) {
                const result = await resultPromise;
                resultPromise = iterator.next();
                if (result.done) break;
                const batch = result.value;

                const now = Date.now();
                const isLogStep = now - lastLog >= logInterval;
                if (isLogStep) {
                    lastLog = now;
                }
                const keepGrads = (options?.metrics?.includes('gradientStatistics') || false) && isLogStep;

                // Do the actual training step
                const lossScalar = this.trainStep(state, batch, false, keepGrads);

                if (options.debug) {
                    const lossValue = (await lossScalar.data())[0];
                    if (isNaN(lossValue) || !isFinite(lossValue)) {
                        console.error('Invalid loss value:', lossValue);
                        console.error('Batch xs:', await batch.xs.array());
                        console.error('Batch ys:', await batch.ys.array());
                        console.error('State:', state);
                        throw new Error('Loss is NaN or Infinity');
                    } else {
                        console.log(`Step ${state.step}: Loss = ${lossValue}`);
                    }
                }

                batch.xs.dispose();
                batch.ys.dispose();

                state.step++;
                state.totalSteps++;

                const willEnd = state.step >= maxSteps;

                if (isLogStep || willEnd) {
                    await this.performLogging(lossScalar, batch.xs.shape[0], evaluator, onStep);
                } else {
                    if (state.gradientNorm) {
                        state.gradientNorm.dispose();
                        state.gradientNorm = undefined;
                    }
                    if (state.accuracy) {
                        state.accuracy.dispose();
                        state.accuracy = undefined;
                    }
                }
                lossScalar.dispose();

                if (willEnd) {
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

        this.model.metaData.actionLog = this.model.metaData.actionLog || [];
        const endTime = Date.now();
        this.model.metaData.actionLog.push({
            action: 'pretrain',
            timestamp: endTime,
            duration: endTime - startTime,
            tokensProcessed: this.tokensProcessed,
            options,
        });

        return { losses: state.losses, validationLosses: state.validationLosses };
    }
}
