import { describe, expect, it } from 'vitest';
import { getTokenConfidence } from './utilities';

describe('inference utilities', () => {
    it('returns 1.0 for a fully certain distribution', () => {
        const confidence = getTokenConfidence([1, 0, 0, 0]);
        expect(confidence).toBeCloseTo(1, 6);
    });

    it('returns 0.0 for a uniform distribution', () => {
        const confidence = getTokenConfidence([0.25, 0.25, 0.25, 0.25]);
        expect(confidence).toBeCloseTo(0, 6);
    });

    it('returns a bounded value for a mixed distribution', () => {
        const confidence = getTokenConfidence([0.7, 0.2, 0.1]);
        expect(confidence).toBeGreaterThan(0);
        expect(confidence).toBeLessThan(1);
    });

    it('returns 0 for an empty distribution', () => {
        const confidence = getTokenConfidence([]);
        expect(confidence).toBe(0);
    });
});
