import { afterEach, describe, it } from 'vitest';
import Generator from './Generator';
import NanoGPT from './NanoGPTModel';
import CharTokeniser from './tokeniser/CharTokeniser';
import * as tf from '@tensorflow/tfjs';

const CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'];

describe('Generator', () => {
    afterEach(() => {
        tf.disposeVariables();
    });

    it('should generate text based on a prompt', async ({ expect }) => {
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
            dropout: 0.1, // Example dropout rate
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const prompt = 'abcde';
        const output = await generator.generate(prompt, { maxLength: 50 });
        expect(output).toBeDefined();
        expect(typeof output).toBe('string');
        expect(output.length).toBeGreaterThan(prompt.length);
        expect(output).toContain(prompt);
    });

    it('can handle an untrained tokeniser', async ({ expect }) => {
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
            dropout: 0.1, // Example dropout rate
        });
        const tokeniser = new CharTokeniser(20);
        const generator = new Generator(model, tokeniser);

        const prompt = 'abcde';
        const output = await generator.generate(prompt, { maxLength: 50 });
        console.log('Output with untrained tokeniser:', output);
        expect(output).toBeDefined();
        expect(typeof output).toBe('string');
        expect(output.length).toBeGreaterThan(prompt.length);
        expect(output).toContain(prompt);
    });

    it('should emit tokens during generation', async ({ expect }) => {
        const model = new NanoGPT({
            vocabSize: 20,
            nEmbed: 64,
            nLayer: 1,
            nHead: 4,
            blockSize: 32,
            dropout: 0.1,
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const emittedTokens: number[][] = [];
        generator.on('tokens', (tokens) => {
            emittedTokens.push(tokens);
        });

        const prompt = 'abcde';
        await generator.generate(prompt, { maxLength: 10 });

        expect(emittedTokens.length).toBeGreaterThan(0);
        expect(emittedTokens[0].length).toBeGreaterThan(0);
    });

    it('should emit tokens with attention when requested', async ({ expect }) => {
        const model = new NanoGPT({
            vocabSize: 20,
            nEmbed: 64,
            nLayer: 1,
            nHead: 2,
            blockSize: 32,
            dropout: 0.1,
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const emittedAttention: number[][][][] = [];
        const emittedTokens: number[][] = [];
        generator.on('tokens', (tokens, _1, attention) => {
            emittedTokens.push(tokens);
            if (attention) {
                emittedAttention.push(attention);
            }
        });

        const prompt = 'abcde';
        await generator.generate(prompt, { maxLength: 10, attentionScores: true });

        expect(emittedAttention).toHaveLength(emittedTokens.length);
        expect(emittedAttention[0]).toHaveLength(emittedTokens[0].length);

        // When cache is used the attention output is full block size.
        expect(emittedAttention[0][0][0][0]).toHaveLength(model.config.gpt.blockSize);
    });

    it('emits attention with RoPE', async ({ expect }) => {
        const model = new NanoGPT({
            vocabSize: 20,
            nEmbed: 64,
            nLayer: 1,
            nHead: 2,
            blockSize: 32,
            dropout: 0.1,
            useRope: true,
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const emittedAttention: number[][][][] = [];
        const emittedTokens: number[][] = [];
        generator.on('tokens', (tokens, _1, attention) => {
            emittedTokens.push(tokens);
            if (attention) {
                emittedAttention.push(attention);
            }
        });

        const prompt = 'abcde';
        await generator.generate(prompt, { maxLength: 10, attentionScores: true, noCache: true });

        expect(emittedAttention).toHaveLength(emittedTokens.length);
        expect(emittedAttention[0]).toHaveLength(emittedTokens[0].length);
        expect(emittedAttention[0][0][0][0]).toHaveLength(prompt.length);
    });

    it('should emit probabilities when requested', async ({ expect }) => {
        const model = new NanoGPT({
            vocabSize: 20,
            nEmbed: 64,
            nLayer: 1,
            nHead: 2,
            blockSize: 32,
            dropout: 0.1,
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const emittedProbabilities: number[][][] = [];
        generator.on('tokens', (_tokens, _text, _attention, probabilities) => {
            if (probabilities) {
                emittedProbabilities.push(probabilities);
            }
        });

        const prompt = 'abcde';
        await generator.generate(prompt, { maxLength: 10, includeProbabilities: true });

        expect(emittedProbabilities.length).toBeGreaterThan(0);
        expect(emittedProbabilities[0].length).toBeGreaterThan(0);
    });
});
