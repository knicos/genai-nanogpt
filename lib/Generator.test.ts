import { afterEach, describe, it } from 'vitest';
import Generator from './Generator';
import NanoGPT from './models/NanoGPTV1';
import CharTokeniser from './tokeniser/CharTokeniser';
import * as tf from '@tensorflow/tfjs';
import { Conversation } from './main';

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

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        const output = await generator.generate(prompt, { maxLength: 50 });
        expect(output).toBeDefined();
        expect(output).toHaveLength(2);
        expect(output[0].content).toContain(prompt[0].content);
        expect(output[1].role).toBe('assistant');
        expect(output[1].content.length).toBeGreaterThan(0);
    });

    it('generates from an empty conversation', async ({ expect }) => {
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

        const prompt: Conversation[] = [];
        const output = (await generator.generate(prompt, { maxLength: 50 })) as Conversation[];
        expect(output).toBeDefined();
        expect(output).toHaveLength(1);
        expect(output[0].role).toBe('assistant');
        expect(output[0].content.length).toBeGreaterThan(0);
    });

    it('generates from a user conversation', async ({ expect }) => {
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

        const prompt: Conversation[] = [{ role: 'user', content: 'hello there' }];
        const output = (await generator.generate(prompt, { maxLength: 50 })) as Conversation[];
        expect(output).toBeDefined();
        expect(output).toHaveLength(2);
        expect(output[1].role).toBe('assistant');
        expect(output[1].content.length).toBeGreaterThan(0);
    });

    it('generates from a long user conversation', async ({ expect }) => {
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

        const prompt: Conversation[] = [{ role: 'user', content: 'hello there' }];
        let output = (await generator.generate(prompt, { maxLength: 50 })) as Conversation[];
        output.push({ role: 'user', content: 'how are you' });
        output = (await generator.generate(output, { maxLength: 50 })) as Conversation[];

        expect(output).toBeDefined();
        expect(output).toHaveLength(4);
        expect(output[3].role).toBe('assistant');
        expect(output[3].content.length).toBeGreaterThan(0);
    });

    it('appends to end of conversation', async ({ expect }) => {
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

        const prompt: Conversation[] = [{ role: 'user', content: 'hello there' }];
        let output = (await generator.generate(prompt, { maxLength: 50 })) as Conversation[];
        //output.push({ role: 'user', content: 'how are you' });
        output = (await generator.generate(output, { maxLength: 50 })) as Conversation[];

        expect(output).toBeDefined();
        expect(output).toHaveLength(3);
        expect(output[2].role).toBe('assistant');
        expect(output[2].content.length).toBe(50);
    });

    it('supports topP', async ({ expect }) => {
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

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        const output = await generator.generate(prompt, { maxLength: 50, topP: 0.8 });
        expect(output).toBeDefined();
        expect(output).toHaveLength(2);
        expect(output[0].content).toContain(prompt[0].content);
        expect(output[1].role).toBe('assistant');
        expect(output[1].content.length).toBeGreaterThan(0);
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

        tokeniser.vocab[1] = '#'; // Manually set unk token for testing

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        const output = await generator.generate(prompt, { maxLength: 50, allowSpecial: true });
        console.log('Output with untrained tokeniser:', output);
        expect(output).toBeDefined();
        expect(output).toHaveLength(2);
        expect(output[0].content).toContain(prompt[0].content);
        expect(output[1].role).toBe('assistant');
        expect(output[1].content.length).toBeGreaterThan(0);
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

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
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

        const emittedTokens: number[][] = [];
        generator.on('tokens', (tokens) => {
            emittedTokens.push(tokens);
        });

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        await generator.generate(prompt, { maxLength: 10, attentionScores: true });

        const emittedAttention = generator.getAttentionData();

        expect(emittedAttention).toHaveLength(emittedTokens.length);
        expect(emittedAttention[0]).toHaveLength(emittedTokens[0].length);

        // When cache is used the attention output is full block size.
        expect(emittedAttention[0][0][0][0]).toHaveLength(model.config.blockSize);
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

        const emittedTokens: number[][] = [];
        generator.on('tokens', (tokens) => {
            emittedTokens.push(tokens);
        });

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        await generator.generate(prompt, { maxLength: 10, attentionScores: true, noCache: true });

        const emittedAttention = generator.getAttentionData();

        expect(emittedAttention).toHaveLength(emittedTokens.length);
        expect(emittedAttention[0]).toHaveLength(emittedTokens[0].length);
        expect(emittedAttention[0][0][0][0]).toHaveLength(32);
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

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        await generator.generate(prompt, { maxLength: 10, includeProbabilities: true });

        const emittedProbabilities = generator.getProbabilitiesData();

        expect(emittedProbabilities.length).toBeGreaterThan(0);
        expect(emittedProbabilities[0].length).toBeGreaterThan(0);
    });
});
