import { afterAll, afterEach, describe, it, vi } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import { create, globals } from 'webgpu';
import { selectBackend } from '@base/backend';
import NanoGPT from '../models/NanoGPTV1';
import CharTokeniser from '../tokeniser/CharTokeniser';
import Training from './training';
import type { TrainingOptions } from '@base/training/types';
import { createTokenStore } from '@base/training/tasks/TokenStore';

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
        blockSize: 8,
    });
}

function createTraining() {
    const model = createModel();
    const tokeniser = new CharTokeniser(CHARS);
    return { model, tokeniser, training: new Training(model, tokeniser) };
}

async function createStore(tokeniser: CharTokeniser, name: string, tokens: Uint16Array) {
    const store = await createTokenStore(name, tokeniser.id, `${name}-dataset`, { noOPFS: true });
    store.appendShard(tokens);
    await store.finish();
    return store;
}

function createOptions(overrides: Partial<TrainingOptions> = {}): TrainingOptions {
    return {
        method: { type: 'pretraining' },
        batchSize: 2,
        learningRate: 1e-3,
        maxEpochs: 1,
        logInterval: 1,
        ...overrides,
    };
}

describe('Training API', () => {
    afterEach(() => {
        tf.disposeVariables();
    });

    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('creates a real training job and marks it done after completion', { timeout: 10000 }, async ({ expect }) => {
        await selectBackend('webgpu');
        const { model } = createTraining();
        const tokeniser = new CharTokeniser(CHARS);
        const store = await createStore(
            tokeniser,
            'training-store-1',
            new Uint16Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        );

        const options = createOptions();
        const datasets = [{ id: 'ds-1', name: 'Dataset 1', conversational: false }];

        const realTraining = new Training(model, tokeniser);
        const job = await realTraining.job(options, store, datasets, undefined);

        expect(typeof job.id).toBe('string');
        expect(job.state).toBe('running');
        expect(job.totalTokens).toBe(10);
        expect(job.trainer).toBeDefined();
        expect(realTraining.getJob(job.id)).toBe(job);

        await vi.waitFor(() => {
            expect(realTraining.getJob(job.id)?.state).toBe('completed');
        }, 10000);

        await store.dispose();
        model.dispose();
    });

    it('emits done and supports listener removal with off()', async ({ expect }) => {
        await selectBackend('webgpu');
        const { model } = createTraining();
        const tokeniser = new CharTokeniser(CHARS);
        const training = new Training(model, tokeniser);

        const doneListener = vi.fn();
        training.on('completed', doneListener);

        const firstStore = await createStore(
            tokeniser,
            'training-store-2',
            new Uint16Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        );
        const firstJob = await training.job(
            createOptions(),
            firstStore,
            [{ id: 'ds-1', name: 'Dataset 1', conversational: false }],
            undefined
        );

        await vi.waitFor(() => {
            expect(training.getJob(firstJob.id)?.state).toBe('completed');
        });

        expect(doneListener).toHaveBeenCalledWith(firstJob.id);

        training.off('completed', doneListener);

        const secondStore = await createStore(
            tokeniser,
            'training-store-3',
            new Uint16Array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        );
        const secondJob = await training.job(
            createOptions(),
            secondStore,
            [{ id: 'ds-2', name: 'Dataset 2', conversational: false }],
            undefined
        );

        await vi.waitFor(() => {
            expect(training.getJob(secondJob.id)?.state).toBe('completed');
        });

        expect(doneListener).toHaveBeenCalledTimes(1);

        await firstStore.dispose();
        await secondStore.dispose();
        model.dispose();
    });

    it('can cancel a real job and remove it from storage', async ({ expect }) => {
        await selectBackend('webgpu');
        const { model } = createTraining();
        const tokeniser = new CharTokeniser(CHARS);
        const training = new Training(model, tokeniser);
        const store = await createStore(
            tokeniser,
            'training-store-4',
            new Uint16Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        );

        const job = await training.job(
            createOptions({ maxEpochs: 10 }),
            store,
            [{ id: 'ds-1', name: 'Dataset 1', conversational: false }],
            undefined
        );

        training.cancel(job.id);
        expect(job.state).toBe('cancelling');

        await vi.waitFor(() => {
            expect(training.getJob(job.id)?.state).toBe('cancelled');
        }, 10000);

        await store.dispose();
        model.dispose();
    });

    it(
        'allows a cancelled job to be resumed later and continue from the same state',
        { timeout: 10000 },
        async ({ expect }) => {
            await selectBackend('webgpu');
            const { model } = createTraining();
            const tokeniser = new CharTokeniser(CHARS);
            const training = new Training(model, tokeniser);

            const tokens = new Uint16Array(Array.from({ length: 160 }, (_, i) => (i % 20) + 1));
            const store = await createStore(tokeniser, 'training-store-cancel-resume', tokens);

            const job = await training.job(
                createOptions({ maxEpochs: 3, logInterval: 1 }),
                store,
                [{ id: 'ds-cancel', name: 'Dataset Cancel Resume', conversational: false }],
                undefined
            );

            await vi.waitFor(() => {
                expect(job.trainer.tokensProcessed).toBeGreaterThan(0);
            });

            const processedBeforeCancel = job.trainer.tokensProcessed;

            training.cancel(job.id);

            await vi.waitFor(() => {
                expect(training.getJob(job.id)?.state).toBe('cancelled');
            });

            // Future behavior target: cancel should keep resumable state rather than deleting the job.
            expect(training.getJob(job.id)).not.toBeNull();

            training.job({ ...job.options, previous_job_id: job.id }, store, job.datasets, undefined);

            await vi.waitFor(() => {
                const resumed = training.getJob(job.id);
                expect(resumed).not.toBeNull();
                expect(resumed?.trainer.tokensProcessed).toBeGreaterThan(processedBeforeCancel);
            });

            await vi.waitFor(() => {
                expect(training.getJob(job.id)?.state).toBe('completed');
            }, 10000);

            await store.dispose();
            model.dispose();
        }
    );

    it('emits progress events according to the configured logging interval', async ({ expect }) => {
        await selectBackend('webgpu');
        const { model } = createTraining();
        const tokeniser = new CharTokeniser(CHARS);
        const training = new Training(model, tokeniser);

        const progressListener = vi.fn();
        training.on('progress', progressListener);

        const tokens = new Uint16Array(Array.from({ length: 240 }, (_, i) => (i % 20) + 1));
        const store = await createStore(tokeniser, 'training-store-progress-events', tokens);

        const options = createOptions({ maxEpochs: 2, logInterval: 1 });

        const job = await training.job(
            options,
            store,
            [{ id: 'ds-progress', name: 'Dataset Progress', conversational: false }],
            undefined
        );

        await vi.waitFor(() => {
            expect(training.getJob(job.id)?.state).toBe('completed');
        });

        // Future behavior target: progress should be emitted at the log interval during training.
        expect(progressListener).toHaveBeenCalled();

        const steps = progressListener.mock.calls
            .map(([eventJob]) => eventJob?.trainer?.log?.at(-1)?.step)
            .filter((step): step is number => typeof step === 'number');

        expect(steps.length).toBeGreaterThan(0);

        await store.dispose();
        model.dispose();
    });

    it('pauses before each log event when pauseOnLog is enabled and resumes on explicit resume()', async ({
        expect,
    }) => {
        await selectBackend('webgpu');
        const { model } = createTraining();
        const tokeniser = new CharTokeniser(CHARS);
        const training = new Training(model, tokeniser);

        const tokens = new Uint16Array(Array.from({ length: 240 }, (_, i) => (i % 20) + 1));
        const store = await createStore(tokeniser, 'training-store-pause-on-log', tokens);

        const options = createOptions({ maxEpochs: 2, logInterval: 2 });

        const job = await training.job(
            options,
            store,
            [{ id: 'ds-pause-log', name: 'Dataset Pause On Log', conversational: false }],
            undefined
        );
        job.breakOnLog.add(1);

        await vi.waitFor(() => {
            expect(job.history).toHaveLength(1);
            expect(job.state).toBe('paused');
        });

        training.resume(job.id);

        await vi.waitFor(() => {
            expect(job.history).toHaveLength(2);
            expect(job.state).toBe('paused');
        });

        await store.dispose();
        model.dispose();
    });
});
