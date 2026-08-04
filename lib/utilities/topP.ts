export default function topP(probs: number[][], tP: number): number[] {
    const sorted = probs[0].map((p, i) => ({ prob: p, index: i })).sort((a, b) => b.prob - a.prob);

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
        const original = probs[0];
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
