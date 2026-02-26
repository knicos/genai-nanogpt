import { TensorStatistics } from '@base/checks/weights';
import { LoRAConfig } from '@base/models/config';
import { NamedTensorMap, Tensor } from '@tensorflow/tfjs-core';

export interface Metrics {
    accuracy?: number;
    perplexity?: number;
    loss: number;
}

export interface TrainingLogEntry {
    trainingMetrics: Metrics;
    validationMetrics?: Metrics;
    step: number;
    time: number;
    example?: string;
    batchSize: number;
    gradientMetrics?: Map<string, TensorStatistics>;
    learningRate?: number;
    gradientNorm?: number;
    weightNorm?: number;
    weightStatistics?: Map<string, TensorStatistics>;
    memoryUsage?: number;
    tokensPerSecond?: number;
    duration: number;
    totalSamples: number;
    samplesPerSecond: number;
}

export interface LRSchedulerConfig {
    warmupSteps: number;
    decayEpochs: number;
    minLearningRate: number;
    epochSteps: number;
}

export interface AdamWOptimizerConfig extends LRSchedulerConfig {
    learningRate: number;
    beta1: number;
    beta2: number;
    epsilon?: number;
    weightDecay: number;
    lossScaling: number;
    clipNorm?: number;
    orthoGrad?: boolean;
}

export type TrainingMetrics =
    | 'accuracy'
    | 'perplexity'
    | 'gradientNorm'
    | 'gradientStatistics'
    | 'weightNorm'
    | 'weightStatistics'
    | 'memoryUsage'
    | 'tokensPerSecond'
    | 'learningRate';

export interface TrainingOptions extends Partial<AdamWOptimizerConfig> {
    batchSize: number; // Batch size for training
    maxEpochs?: number; // Maximum number of epochs
    logInterval?: number; // Interval for logging training progress
    prompt?: string; // Prompt for generating text during training
    validationSplit?: number; // Fraction of data to use for validation
    gradientCheckpointing?: boolean; // Whether to use gradient checkpointing
    mixedPrecision?: boolean; // Whether to use mixed precision training
    trainableWeights?: string[]; // List of weight names to train (supports glob patterns)
    loraConfig?: LoRAConfig; // LoRA configuration for training
    sftMode: 'full' | 'lora' | 'last-layer'; // Mode for SFT training, if applicable
    maskedLoss?: boolean; // Whether to use masked loss (e.g., for language modeling)
    metrics?: TrainingMetrics[]; // Metrics to compute during training
    contextScaling?: number; // Factor to scale the context length for training (e.g., 0.5 to use half the context length)
    onStep?: (log: TrainingLogEntry) => void; // Callback for each training step
}

export interface TrainingState {
    step: number;
    lastLoss: number;
    totalSteps: number;
    losses: number[];
    validationLosses: number[];
    logStartTime: number;
    trainingDuration: number;
    gradients?: NamedTensorMap;
    gradientNorm?: Tensor;
    accuracy?: Tensor;
}
