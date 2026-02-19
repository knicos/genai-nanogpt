import { describe, it, afterEach, afterAll } from 'vitest';
import '@tensorflow/tfjs';
import * as tf from '@tensorflow/tfjs-core';
import { create, globals } from 'webgpu';
import { selectBackend } from '@base/backend';
import LoRA from './LoRA';
import WeightStore from './WeightStore';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

describe('LoRA', { timeout: 10000 }, () => {
    afterEach(() => {
        tf.disposeVariables();
    });
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('applies LoRA delta on read when attached (no merge)', async ({ expect }) => {
        await selectBackend('webgpu');

        const weightStore = new WeightStore();
        const outDim = 2;
        const inDim = 3;
        const rank = 1;
        const alpha = 5;

        const base = tf.variable(tf.zeros([outDim, inDim]), true, 'w');
        weightStore.addVariable('w', base);

        const lora = new LoRA(weightStore, alpha, rank, ['w']);

        const loraA = weightStore.getRawVariable('w_loraA');
        const loraB = weightStore.getRawVariable('w_loraB');

        loraA.assign(tf.ones([outDim, rank]));
        loraB.assign(tf.ones([rank, inDim]));

        const adjusted = weightStore.getVariable('w');
        const adjustedData = await adjusted.data();

        expect(adjustedData.every((v) => Math.abs(v - alpha) < 1e-6)).toBe(true);

        adjusted.dispose();
        lora.dispose();
    });

    it('merges LoRA update into base weights correctly', async ({ expect }) => {
        await selectBackend('webgpu');

        const weightStore = new WeightStore();
        const outDim = 3;
        const inDim = 2;
        const rank = 2;
        const alpha = 4;

        const base = tf.variable(tf.zeros([outDim, inDim]), true, 'w');
        weightStore.addVariable('w', base);

        const lora = new LoRA(weightStore, alpha, rank, ['w']);

        const loraA = weightStore.getRawVariable('w_loraA');
        const loraB = weightStore.getRawVariable('w_loraB');

        loraA.assign(tf.ones([outDim, rank]));
        loraB.assign(tf.ones([rank, inDim]));

        lora.merge();

        const merged = weightStore.getRawVariable('w');
        const mergedData = await merged.data();

        expect(mergedData.every((v) => Math.abs(v - alpha) < 1e-6)).toBe(true);

        lora.dispose();
    });

    it('detach(true) keeps merged weights and removes LoRA variables', async ({ expect }) => {
        await selectBackend('webgpu');

        const weightStore = new WeightStore();
        const outDim = 2;
        const inDim = 2;
        const rank = 2;
        const alpha = 6;

        const base = tf.variable(tf.zeros([outDim, inDim]), true, 'w');
        weightStore.addVariable('w', base);

        const lora = new LoRA(weightStore, alpha, rank, ['w']);

        const loraA = weightStore.getRawVariable('w_loraA');
        const loraB = weightStore.getRawVariable('w_loraB');

        loraA.assign(tf.ones([outDim, rank]));
        loraB.assign(tf.ones([rank, inDim]));

        const adjusted = weightStore.getVariable('w');
        const adjustedData = await adjusted.data();
        expect(adjustedData.every((v) => Math.abs(v - alpha) < 1e-6)).toBe(true);
        adjusted.dispose();

        lora.detach(true);

        expect(weightStore.hasVariable('w_loraA')).toBe(false);
        expect(weightStore.hasVariable('w_loraB')).toBe(false);

        const merged = weightStore.getRawVariable('w');
        const mergedData = await merged.data();
        expect(mergedData.every((v) => Math.abs(v - alpha) < 1e-6)).toBe(true);

        lora.dispose();
    });
});
