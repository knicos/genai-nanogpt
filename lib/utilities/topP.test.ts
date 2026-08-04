import { describe, it } from 'vitest';
import topP from './topP';

describe('topP', () => {
    it('returns correct top-p probabilities for a simple distribution', ({ expect }) => {
        const probs = [[0.1, 0.2, 0.3, 0.4]];
        const tP = 0.5;
        const result = topP(probs, tP);
        const expected = [0, 0, 0.42857142857142855, 0.5714285714285714];
        expected.forEach((v, i) => expect(result[i]).toBeCloseTo(v, 12));
        const sum = result.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1);
    });

    it('with tP=1 returns the original normalized probabilities', ({ expect }) => {
        const probs = [[0.1, 0.2, 0.3, 0.4]];
        const tP = 1;
        const result = topP(probs, tP);
        const expected = [0.1, 0.2, 0.3, 0.4];
        expected.forEach((v, i) => expect(result[i]).toBeCloseTo(v, 12));
        const sum = result.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1);
    });

    it('with tP=0 returns only the single highest-probability token', ({ expect }) => {
        const probs = [[0.1, 0.2, 0.3, 0.4]];
        const tP = 0;
        const result = topP(probs, tP);
        expect(result).toEqual([0, 0, 0, 1]);
    });

    it('handles ties correctly and renormalizes included tokens', ({ expect }) => {
        const probs = [[0.25, 0.25, 0.25, 0.25]];
        const tP = 0.5;
        const result = topP(probs, tP);
        const expected = [0.5, 0.5, 0, 0];
        expected.forEach((v, i) => expect(result[i]).toBeCloseTo(v, 12));
    });

    it('falls back to a uniform distribution when all probabilities are zero', ({ expect }) => {
        const probs = [[0, 0, 0, 0]];
        const tP = 0.5;
        const result = topP(probs, tP);
        const expected = [0.25, 0.25, 0.25, 0.25];
        expected.forEach((v, i) => expect(result[i]).toBeCloseTo(v, 12));
        const sum = result.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1);
    });
});
