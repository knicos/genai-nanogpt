import BaseLayer from '@base/layers/BaseLayer';
import { Tensor } from '@tensorflow/tfjs-core';

export interface TensorStatistics {
    mean: number;
    std: number;
    min: number;
    max: number;
    sparsity: number; // Percentage of zero elements
    isFinite: boolean;
    hasNaN: boolean;
    closeToZeroCount: number; // Count of elements close to zero
}

export async function createTensorStatistics(weight: Tensor | number[]): Promise<TensorStatistics> {
    if (!Array.isArray(weight)) {
        if (weight.dtype !== 'float32') {
            throw new Error(`Unsupported dtype ${weight.dtype} for weight statistics.`);
        }
    }
    const data = Array.isArray(weight) ? weight : await weight.data();
    const size = data.length;

    let sum = 0;
    let sumSq = 0;
    let min = data[0];
    let max = data[0];
    let zeroCount = 0;
    let isFinite = true;
    let hasNaN = false;
    let closeToZeroCount = 0;

    for (let i = 0; i < size; i++) {
        const value = data[i];
        sum += value;
        sumSq += value * value;
        if (value < min) min = value;
        if (value > max) max = value;
        if (value === 0) zeroCount++;
        if (Math.abs(value) < 1e-8) closeToZeroCount++;
        if (!isFinite && isFinite !== false) {
            isFinite = Number.isFinite(value);
        }
        if (Number.isNaN(value)) {
            hasNaN = true;
        }
    }

    const mean = sum / size;
    const variance = sumSq / size - mean * mean;
    const std = Math.sqrt(variance);
    const sparsity = zeroCount / size;

    return {
        mean,
        std,
        min,
        max,
        sparsity,
        isFinite,
        hasNaN,
        closeToZeroCount,
    };
}

export async function createWeightStatistics(layer: BaseLayer): Promise<{ [key: string]: TensorStatistics }> {
    const weights = layer.trainableVariables;
    const stats: { [key: string]: TensorStatistics } = {};

    for (const weight of weights) {
        stats[weight.name] = await createTensorStatistics(weight);
    }

    return stats;
}
