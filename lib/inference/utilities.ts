import { SPECIALS } from '@base/tokeniser/BaseTokeniser';
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

export const CHARS = [
    ...SPECIALS,
    ...Array.from({ length: 95 }, (_, i) => String.fromCharCode(i + 32)), // ASCII
    // Spanish accented letters and punctuation
    ...'áéíóúüñ¿¡',
    // Finnish accented letters
    ...'äöÄÖÅå',
    // Greek letters
    ...'αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ',
    // Cyrillic letters
    ...'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',
];

export function padArray(arr: string[], length: number): string[] {
    if (arr.length === length) return arr;
    if (arr.length > length) return arr.slice(0, length);
    return arr.concat(Array(length - arr.length).fill(''));
}
