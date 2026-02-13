import { TensorStatistics } from '@base/checks/weights';
import { NamedTensorMap } from '@tensorflow/tfjs-core';

export interface TrainingLogEntry {
    loss: number;
    valLoss?: number;
    step: number;
    time: number;
    example?: string;
    batchSize: number;
    gradientMetrics?: Map<string, TensorStatistics>;
    learningRate?: number;
}

export interface TrainingProgress {
    duration: number;
    totalSamples: number;
    samplesPerSecond: number;
    memory?: number;
}

export interface AdamConfig {
    learningRate: number;
    minLearningRate?: number;
    weightDecay?: number;
    warmupSteps?: number;
    decaySteps?: number;
    lossScaling: number;
    beta1: number;
    beta2: number;
    epsilon: number;
}

export interface TrainingOptions {
    desiredLoss: number;
    logInterval: number;
    prompt?: string;
    maxSteps: number;
    advancedMetrics?: boolean;
    gradientMetrics?: boolean;
    maskedLoss?: boolean;
    onStep?: (log: TrainingLogEntry, progress: TrainingProgress) => Promise<void> | void;
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
}
