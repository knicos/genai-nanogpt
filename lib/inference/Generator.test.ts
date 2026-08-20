import { afterAll, afterEach, describe, it } from 'vitest';
import Generator from './Generator';
import NanoGPT from '../models/NanoGPTV1';
import CharTokeniser from '../tokeniser/CharTokeniser';
import * as tf from '@tensorflow/tfjs';
import { Conversation } from '../main';
import { create, globals } from 'webgpu';
import { selectBackend } from '@base/backend';
import { GeneratorConversation, IGeneratorOutput } from './types';
import arrayShape from '../utilities/arrayShape';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

const CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'];

describe('Generator', () => {
    afterEach(() => {
        tf.disposeVariables();
    });
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('should generate text based on a prompt', { timeout: 10000 }, async ({ expect }) => {
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
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
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
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
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
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
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
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
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const prompt: Conversation[] = [{ role: 'user', content: 'hello there' }];
        let output = (await generator.generate(prompt, { maxLength: 50 })) as Conversation[];
        //output.push({ role: 'user', content: 'how are you' });
        (output as GeneratorConversation[])[output.length - 1]._completed = false;
        output = (await generator.generate(output, { maxLength: 50 })) as Conversation[];

        expect(output).toBeDefined();
        expect(output).toHaveLength(2);
        expect(output[1].role).toBe('assistant');
        expect(output[1].content.length).toBe(100);
    });

    it('appends new conversation', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
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
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
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
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
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
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20,
            nEmbed: 64,
            nLayer: 1,
            nHead: 4,
            blockSize: 32,
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const emittedTokens: IGeneratorOutput[] = [];
        generator.on('tokens', (tokens) => {
            emittedTokens.push(tokens);
        });

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        await generator.generate(prompt, { maxLength: 10 });

        expect(emittedTokens.length).toBeGreaterThan(0);
    });

    it('should emit tokens with attention when requested', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20,
            nEmbed: 64,
            nLayer: 1,
            nHead: 2,
            blockSize: 32,
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const emittedTokens: IGeneratorOutput[] = [];
        generator.on('tokens', (tokens) => {
            emittedTokens.push(tokens);
        });

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        await generator.generate(prompt, { maxLength: 10, outputAttention: true });

        const output = generator.getRawOutput();
        const emittedAttention = output.map((o) => o.attention).filter((a) => a !== null);

        expect(emittedAttention).toHaveLength(emittedTokens.length);

        // When cache is used the attention output is full block size.
        expect(emittedAttention[0][0][0][0]).toHaveLength(model.config.blockSize);
    });

    it('emits attention with RoPE', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20,
            nEmbed: 64,
            nLayer: 3,
            nHead: 2,
            blockSize: 32,
            useRope: true,
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const emittedTokens: IGeneratorOutput[] = [];
        generator.on('tokens', (tokens) => {
            emittedTokens.push(tokens);
        });

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        await generator.generate(prompt, { maxLength: 10, outputAttention: true, noCache: true });

        const output = generator.getRawOutput();
        const emittedAttention = output.map((o) => o.attention).filter((a) => a !== null);

        const attentionShape = arrayShape(emittedAttention);

        expect(attentionShape).toEqual([10, 3, 2, 1, 32]); // [tokens, layers, heads, tokens per step, blockSize]
    });

    it('should emit probabilities when requested', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20,
            nEmbed: 64,
            nLayer: 1,
            nHead: 2,
            blockSize: 32,
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        await generator.generate(prompt, { maxLength: 10, outputScores: true });

        const output = generator.getRawOutput();
        const emittedProbabilities = output.map((o) => o.scores).filter((s) => s !== null);

        const probabilitiesShape = arrayShape(emittedProbabilities);
        expect(probabilitiesShape).toEqual([10, model.config.vocabSize]);
    });

    it('should emit last multinomial random value', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20,
            nEmbed: 64,
            nLayer: 1,
            nHead: 2,
            blockSize: 32,
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        await generator.generate(prompt, { maxLength: 10, outputScores: true });

        const lastMultinomialRand = generator.getRawOutput().slice(-1)[0].multinomialRand;

        expect(lastMultinomialRand).not.toBeNull();
    });

    it('can attach a LoRA', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
        });

        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        // Dummy pass
        await generator.generate(prompt, { maxLength: 50 });

        model.createLoRA('test-lora', {
            rank: 4,
            alpha: 8,
            variables: ['*'],
        });

        const output = await generator.generate(prompt, { maxLength: 50, loraName: 'test-lora' });
        expect(output).toBeDefined();
        expect(output).toHaveLength(3);
        expect(output[0].content).toContain(prompt[0].content);
        expect(output[2].role).toBe('assistant');
        expect(output[2].content.length).toBeGreaterThan(0);
        expect(model.hasLoRA()).toBe(true);
        expect(model.lora?.name).toBe('test-lora');
    });

    it('should queue generator jobs', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = new NanoGPT({
            vocabSize: 20, // Example vocab size
            nEmbed: 64, // Example embedding size
            nLayer: 1, // Example number of layers
            nHead: 2, // Example number of attention heads
            blockSize: 32, // Example block size
        });
        const tokeniser = new CharTokeniser(CHARS);
        const generator = new Generator(model, tokeniser);

        const prompt: Conversation[] = [{ role: 'user', content: 'abcde' }];
        const promises = [
            generator.generate(prompt, { maxLength: 50 }),
            generator.generate(prompt, { maxLength: 50 }),
            generator.generate(prompt, { maxLength: 50 }),
        ];
        expect(generator.getQueueLength()).toBe(2); // 1 running, 2 queued
        const results = await Promise.all(promises);
        expect(results).toHaveLength(3);
    });
});
