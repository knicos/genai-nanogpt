import { afterAll, afterEach, describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { create, globals } from 'webgpu';
import { selectBackend } from '@base/backend';
import NanoGPT from '../models/NanoGPTV1';
import CharTokeniser from '../tokeniser/CharTokeniser';
import type { Conversation } from '../tokeniser/type';
import Beamer from './Beamer';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

const CHARS = [
    ' ',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'i',
    'j',
    'k',
    'l',
    'm',
    'n',
    'o',
    'p',
    'q',
    'r',
    's',
    't',
    'u',
    'v',
    'w',
    'x',
    'y',
    'z',
];

function createSmallModel() {
    return new NanoGPT({
        vocabSize: CHARS.length,
        nEmbed: 64,
        nLayer: 1,
        nHead: 2,
        blockSize: 32,
    });
}

describe('Beamer', () => {
    afterEach(() => {
        tf.disposeVariables();
    });

    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('returns up to N ranked completions with bounded token length', { timeout: 10000 }, async ({ expect }) => {
        await selectBackend('webgpu');
        const model = createSmallModel();
        const tokeniser = new CharTokeniser(CHARS);
        const beamer = new Beamer(model, tokeniser);

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        const result = await beamer.beam(prompt, { beams: 4, maxBeamLength: 3, topP: 1, noCache: true });

        expect(result.length).toBeGreaterThan(0);
        expect(result.length).toBeLessThanOrEqual(4);

        for (let i = 0; i < result.length; i++) {
            const beam = result[i];
            expect(beam.tokens.length).toBeLessThanOrEqual(3);
            expect(beam.tokens.length).toBeGreaterThan(0);
            expect(Number.isFinite(beam.score)).toBe(true);
            expect(beam.text).toBe(tokeniser.decode(beam.tokens));

            if (i > 0) {
                expect(result[i - 1].score).toBeGreaterThanOrEqual(beam.score);
            }
        }
    });

    it('handles restrictive topP by returning fewer candidates when probability mass is narrow', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = createSmallModel();
        const tokeniser = new CharTokeniser(CHARS);
        const beamer = new Beamer(model, tokeniser);

        const prompt: Conversation[] = [{ role: 'user', content: 'hello there' }];
        const result = await beamer.beam(prompt, { beams: 5, maxBeamLength: 4, topP: 0, noCache: true });

        expect(result.length).toBe(1);
        expect(result[0].tokens.length).toBeGreaterThan(0);
        expect(result[0].tokens.length).toBeLessThanOrEqual(4);
    });

    it('supports untrained tokeniser vocab fallback while still producing beams', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = createSmallModel();
        const tokeniser = new CharTokeniser(CHARS.length);
        const beamer = new Beamer(model, tokeniser);

        tokeniser.vocab[1] = '#';

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        const result = await beamer.beam(prompt, { beams: 3, maxBeamLength: 2, topP: 1, allowSpecial: true });

        expect(result.length).toBeGreaterThan(0);
        expect(result.length).toBeLessThanOrEqual(3);
        expect(result[0].tokens.length).toBeGreaterThan(0);
    });

    it('allows beams to continue past maxBeamLength when endOnWhiteSpace is enabled', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = createSmallModel();
        const tokeniser = new CharTokeniser(CHARS);
        const beamer = new Beamer(model, tokeniser);

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        const minLength = 1;
        const result = await beamer.beam(prompt, {
            beams: 3,
            maxBeamLength: minLength,
            endOnWhiteSpace: true,
            topP: 1,
            noCache: true,
            maxLength: 12,
        });

        expect(result.length).toBeGreaterThan(0);
        expect(result.length).toBeLessThanOrEqual(3);
        expect(result.some((beam) => beam.tokens.length > minLength)).toBe(true);
        expect(result.every((beam) => beam.tokens.length >= minLength)).toBe(true);
    });
});
