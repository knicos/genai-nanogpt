import { afterAll, afterEach, describe, it, vi } from 'vitest';
import NanoGPT from '../models/NanoGPTV1';
import CharTokeniser from '../tokeniser/CharTokeniser';
import * as tf from '@tensorflow/tfjs';
import { create, globals } from 'webgpu';
import { selectBackend } from '@base/backend';
import Responses from './responses';
import { IGeneratorResponse } from '@base/inference/types';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

const CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'];

function createModel() {
    return new NanoGPT({
        vocabSize: 20,
        nEmbed: 64,
        nLayer: 1,
        nHead: 2,
        blockSize: 32,
    });
}

function createResponses(model = createModel()) {
    const tokeniser = new CharTokeniser(CHARS);
    return { responses: new Responses(model, tokeniser), model };
}

describe('Responses API', () => {
    afterEach(() => {
        tf.disposeVariables();
    });
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('can generate a plain text response', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = createModel();
        const tokeniser = new CharTokeniser(CHARS);

        const responses = new Responses(model, tokeniser);

        const response = await responses.create({
            input: 'abc',
            temperature: 0.7,
            topK: 5,
            topP: 0.9,
            maxLength: 10,
        });

        expect(Array.isArray(response.output)).toBe(true);
        expect(response.output?.length).toBe(2);
        expect(response.output?.[1]._output?.length).toBe(10);
        expect(response.done).toBe(true);
        expect(response.output?.[0].role).toBe('user');
        expect(response.output?.[0].content).toBe('abc');
        expect(response.output?.[1].role).toBe('assistant');
        expect(response.output?.[1].content.length).toBe(10);
    });

    it('can generate a background response', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = createModel();
        const tokeniser = new CharTokeniser(CHARS);

        const responses = new Responses(model, tokeniser);

        const response = await responses.create({
            input: 'abc',
            temperature: 0.7,
            topK: 5,
            topP: 0.9,
            maxLength: 10,
            background: true,
        });

        expect(response.output).toBe(null);
        expect(response.done).toBe(false);
        expect(typeof response.id).toBe('string');

        await vi.waitFor(() => {
            const retrievedResponse = responses.getResponse(response.id);
            expect(retrievedResponse).not.toBeNull();
            expect(retrievedResponse?.done).toBe(true);
            expect(Array.isArray(retrievedResponse?.output)).toBe(true);
            expect(retrievedResponse?.output?.length).toBe(2);
            expect(retrievedResponse?.output?.[1]._output?.length).toBe(10);
        });
    });

    it('queues jobs', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = createModel();
        const tokeniser = new CharTokeniser(CHARS);

        const responses = new Responses(model, tokeniser);

        const promise1 = responses.create({
            input: 'abc',
            temperature: 0.7,
            topK: 5,
            topP: 0.9,
            maxLength: 10,
        });

        const promise2 = responses.create({
            input: 'abc',
            temperature: 0.7,
            topK: 5,
            topP: 0.9,
            maxLength: 10,
        });

        expect(responses.queued).toBe(1);

        await promise1;

        expect(responses.queued).toBe(0);

        await promise2;
    });

    it('can output a confidence value', async ({ expect }) => {
        await selectBackend('webgpu');
        const { responses } = createResponses();

        const response = await responses.create({
            input: 'abc',
            temperature: 0.7,
            topK: 5,
            topP: 0.9,
            maxLength: 10,
            allowSpecial: true,
            outputConfidence: true,
        });

        console.log(response.output?.[1]._output);

        expect(Array.isArray(response.output)).toBe(true);
        expect(response.output?.length).toBe(2);
        expect(response.output?.[1]._output?.length).toBe(10);
        expect(response.output?.[1]._output?.[0].confidence).not.toBeNull();
        expect(response.output?.[1]._output?.[0].confidence).toBeGreaterThanOrEqual(0);
        expect(response.output?.[1]._output?.[0].confidence).toBeLessThanOrEqual(1);
    });

    it('supports output toggles for attention, scores, logits and hidden states', async ({ expect }) => {
        await selectBackend('webgpu');
        const { responses } = createResponses();
        const hiddenStateModes: ('embedding' | 'logits' | 'softmax' | 'all')[] = [
            'embedding',
            'logits',
            'softmax',
            'all',
        ];

        for (const outputHiddenStates of hiddenStateModes) {
            const response = await responses.create({
                input: 'abc',
                maxLength: 4,
                allowSpecial: true,
                outputAttention: true,
                outputScores: true,
                outputLogits: true,
                outputHiddenStates,
            });

            const firstTokenOutput = response.output?.[1]._output?.[0];
            expect(firstTokenOutput).toBeDefined();
            expect(firstTokenOutput?.attention).not.toBeNull();
            expect(firstTokenOutput?.scores).not.toBeNull();
            expect(firstTokenOutput?.logits).not.toBeNull();
            expect(firstTokenOutput?.hiddenStates).not.toBeNull();
            expect(firstTokenOutput?.scores?.length).toBe(20);
            expect(firstTokenOutput?.logits?.length).toBe(20);
            expect(firstTokenOutput?.hiddenStates?.length).toBeGreaterThan(0);
        }
    });

    it('supports targets and output loss over generated tokens', async ({ expect }) => {
        await selectBackend('webgpu');
        const { responses } = createResponses();
        const targets = [1, 2, 3, 4, 5];

        const response = await responses.create({
            input: 'abc',
            maxLength: 5,
            allowSpecial: true,
            outputLoss: true,
            targets,
        });

        expect(response.done).toBe(true);
        expect(response.output?.[1]._output?.length).toBe(5);

        const losses = response.output?.[1]._output?.map((o) => o.loss) ?? [];
        expect(losses.some((value) => typeof value === 'number')).toBe(true);
    });

    it('supports non-conversational and continuation output modes', async ({ expect }) => {
        await selectBackend('webgpu');
        const { responses } = createResponses();

        const response = await responses.create({
            input: [{ role: 'text', content: 'seed' }],
            nonConversational: true,
            continuation: true,
            maxLength: 5,
            allowSpecial: true,
        });

        expect(response.output).not.toBeNull();
        expect(response.output?.length).toBe(1);
        expect(response.output?.[0].role).toBe('text');
        expect(response.output?.[0].content.length).toBeGreaterThanOrEqual(4);
        expect(response.output?.[0]._output?.length).toBe(5);
    });

    it('supports previous_response_id to continue a prior conversation', async ({ expect }) => {
        await selectBackend('webgpu');
        const { responses } = createResponses();

        const first = await responses.create({
            input: 'abc',
            maxLength: 3,
        });

        const second = await responses.create({
            previous_response_id: first.id,
            maxLength: 3,
        });

        expect(first.done).toBe(true);
        expect(second.done).toBe(true);
        expect(second.output).not.toBeNull();
        expect(second.output?.length).toBeGreaterThanOrEqual(3);
        expect(second.output?.[0].role).toBe('user');
    });

    it('supports loraName attachment via responses options', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = createModel();
        const { responses } = createResponses(model);

        // Warm up once so model weights are materialized before creating LoRA tensors.
        await responses.create({
            input: 'abc',
            maxLength: 1,
            allowSpecial: true,
        });

        model.createLoRA('test-lora', {
            rank: 4,
            alpha: 8,
            variables: ['*'],
        });

        const response = await responses.create({
            input: 'abc',
            maxLength: 3,
            allowSpecial: true,
            loraName: 'test-lora',
        });

        expect(response.done).toBe(true);
        expect(model.hasLoRA()).toBe(true);
        expect(model.lora?.name).toBe('test-lora');
    });

    it('accepts no-op/pass-through options without failing', async ({ expect }) => {
        await selectBackend('webgpu');
        const { responses } = createResponses();

        const response = await responses.create({
            input: 'abc',
            maxLength: 3,
            temperature: 0.8,
            topK: 5,
            topP: 0.9,
            noCache: true,
            outputConfidence: true,
            outputMultinomialRand: true,
            chunkSize: 2,
        });

        expect(response.done).toBe(true);
        expect(response.output?.[1]._output?.length).toBe(3);
        expect(response.output?.[1]._output?.[0].multinomialRand).not.toBeNull();
    });

    it('generates callbacks', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = createModel();
        const tokeniser = new CharTokeniser(CHARS);

        const responses = new Responses(model, tokeniser);

        const log: IGeneratorResponse[] = [];

        const cb = vi.fn((response: IGeneratorResponse) => {
            log.push(response);
        });

        await responses.create(
            {
                input: 'abc',
                temperature: 0.7,
                topK: 5,
                topP: 0.9,
                maxLength: 10,
            },
            cb
        );

        expect(cb).toHaveBeenCalledTimes(10);
        expect(log).toHaveLength(10);
    });

    it('generates chunk callbacks', async ({ expect }) => {
        await selectBackend('webgpu');
        const model = createModel();
        const tokeniser = new CharTokeniser(CHARS);

        const responses = new Responses(model, tokeniser);

        const log: IGeneratorResponse[] = [];

        const cb = vi.fn((response: IGeneratorResponse) => {
            log.push(response);
        });

        await responses.create(
            {
                input: 'abc',
                temperature: 0.7,
                topK: 5,
                topP: 0.9,
                maxLength: 10,
                chunkSize: 2,
            },
            cb
        );

        expect(cb).toHaveBeenCalledTimes(5);
        expect(log).toHaveLength(5);
    });

    it('blocks generation when a hook is attached until resumed', async ({ expect }) => {
        await selectBackend('webgpu');
        const { responses } = createResponses();

        const response = await responses.create({
            input: 'abc',
            temperature: 0.7,
            topK: 5,
            topP: 0.9,
            maxLength: 10,
            background: true,
        });

        const hooked = responses.hook(response.id);
        expect(hooked).toBe(true);

        let hookCount = 0;
        let autoResume = false;

        responses.on('hook', (id) => {
            if (id !== response.id) return;
            hookCount++;
            if (autoResume) {
                responses.resume(id);
            }
        });

        await vi.waitFor(() => {
            expect(hookCount).toBeGreaterThan(0);
        });

        expect(responses.getResponse(response.id)?.done).toBe(false);

        autoResume = true;
        responses.resume(response.id);

        await vi.waitFor(() => {
            expect(responses.getResponse(response.id)?.done).toBe(true);
        });
    });
});
