export default function topP(probs: number[][] | number[], tP: number): number[] {
    const actualProbs = Array.isArray(probs[0]) ? probs[0] : (probs as number[]);
    const sorted = actualProbs.map((p, i) => ({ prob: p, index: i })).sort((a, b) => b.prob - a.prob);

    let cumulativeProb = 0;
    const masked = new Array<number>(sorted.length).fill(0);
    for (const item of sorted) {
        cumulativeProb += item.prob;
        masked[item.index] = item.prob;
        if (cumulativeProb >= tP) {
            break;
        }
    }

    // Renormalize
    const sumMasked = masked.reduce((a, b) => a + b, 0);

    if (sumMasked === 0) {
        const original = actualProbs;
        const origSum = original.reduce((a, b) => a + b, 0);
        if (origSum > 0) {
            return original.map((p) => p / origSum);
        }
        const uniform = 1 / original.length;
        return original.map(() => uniform);
    }

    const renormProbs = masked.map((p) => p / sumMasked);
    return renormProbs;
}
