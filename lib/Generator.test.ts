import { describe, it } from 'vitest';
import Generator from './Generator';
import NanoGPT from './NanoGPTModel';
import CharTokeniser from './tokeniser/CharTokeniser';
import * as tf from '@tensorflow/tfjs';

describe('Generator', () => {
    it('should generate text based on a prompt', async ({ expect }) => {
        const model = new NanoGPT(tf, {
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
            dropout: 0.1, // Example dropout rate
        });
        const tokeniser = new CharTokeniser([
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
        ]);
        const generator = new Generator(model, tokeniser);

        const prompt = 'abcde';
        const output = await generator.generate(prompt, { maxLength: 50 });
        expect(output).toBeDefined();
        expect(typeof output).toBe('string');
        expect(output.length).toBeGreaterThan(prompt.length);
        expect(output).toContain(prompt);
    });
});
