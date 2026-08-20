import type { IGeneratorOutput } from './types';

/**
 * Confidence derived from normalized entropy of the probability distribution.
 * Returns a value in [0, 1], where 1 is fully confident (delta-like distribution)
 * and 0 is maximally uncertain (uniform distribution).
 */
export function getTokenConfidence(probabilities: number[]): number {
    if (!probabilities.length) {
        return 0;
    }

    let entropy = 0;
    for (const p of probabilities) {
        if (p > 0) {
            entropy -= p * Math.log(p);
        }
    }

    const maxEntropy = Math.log(probabilities.length);
    if (maxEntropy <= 0) {
        return 1;
    }

    const normalizedEntropy = entropy / maxEntropy;
    return Math.min(1, Math.max(0, 1 - normalizedEntropy));
}

export function getAttention(output: IGeneratorOutput, layer: number, head: number): number[] | null {
    if (!output.attention || !output.attention[layer] || !output.attention[layer][head]) {
        return null;
    }
    const numberOfOutputs = output.attention[layer][head].length;
    return output.attention[layer][head][numberOfOutputs - 1];
}

export function getHiddenState(output: IGeneratorOutput, layer: number): number[] | null {
    if (!output.hiddenStates || !output.hiddenStates[layer]) {
        return null;
    }
    return output.hiddenStates[layer];
}
