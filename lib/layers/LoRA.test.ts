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

        const lora = new LoRA('test1', weightStore, alpha, rank, ['w']);
        lora.attach();

        const loraA = weightStore.getRawVariable('w_test1_loraA');
        const loraB = weightStore.getRawVariable('w_test1_loraB');

        loraA.assign(tf.ones([outDim, rank]));
        loraB.assign(tf.ones([rank, inDim]));

        const adjusted = weightStore.getVariable('w');
        const adjustedData = await adjusted.data();

        expect(adjustedData.every((v) => Math.abs(v - alpha) < 1e-6)).toBe(true);

        adjusted.dispose();
        lora.dispose();
    });

    it('supports multiple LoRA instances', async ({ expect }) => {
        await selectBackend('webgpu');

        const weightStore = new WeightStore();
        const outDim = 2;
        const inDim = 3;
        const rank = 1;
        const alpha = 5;

        const base = tf.variable(tf.zeros([outDim, inDim]), true, 'w');
        weightStore.addVariable('w', base);

        const lora1 = new LoRA('test1', weightStore, alpha, rank, ['w']);
        const lora2 = new LoRA('test2', weightStore, alpha, rank, ['w']);
        lora1.attach();

        let loraA = weightStore.getRawVariable('w_test1_loraA');
        let loraB = weightStore.getRawVariable('w_test1_loraB');
        loraA.assign(tf.ones([outDim, rank]));
        loraB.assign(tf.ones([rank, inDim]));
        loraA = weightStore.getRawVariable('w_test2_loraA');
        loraB = weightStore.getRawVariable('w_test2_loraB');
        loraA.assign(tf.ones([outDim, rank]));
        loraB.assign(tf.ones([rank, inDim]));

        const adjusted = weightStore.getVariable('w');
        const adjustedData = await adjusted.data();

        expect(adjustedData.every((v) => Math.abs(v - alpha) < 1e-6)).toBe(true);

        adjusted.dispose();
        lora1.dispose();
        lora2.dispose();
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

        const lora = new LoRA('test2', weightStore, alpha, rank, ['w']);

        const loraA = weightStore.getRawVariable('w_test2_loraA');
        const loraB = weightStore.getRawVariable('w_test2_loraB');

        loraA.assign(tf.ones([outDim, rank]));
        loraB.assign(tf.ones([rank, inDim]));

        lora.merge();

        const merged = weightStore.getRawVariable('w');
        const mergedData = await merged.data();

        expect(mergedData.every((v) => Math.abs(v - alpha) < 1e-6)).toBe(true);

        lora.dispose();
    });

    it('detach() keeps LoRA variables', async ({ expect }) => {
        await selectBackend('webgpu');

        const weightStore = new WeightStore();
        const outDim = 2;
        const inDim = 2;
        const rank = 2;
        const alpha = 6;

        const base = tf.variable(tf.zeros([outDim, inDim]), true, 'w');
        weightStore.addVariable('w', base);

        const lora = new LoRA('test3', weightStore, alpha, rank, ['w']);
        lora.attach();

        expect(weightStore.onWeightRead).toBeDefined();

        const loraA = weightStore.getRawVariable('w_test3_loraA');
        const loraB = weightStore.getRawVariable('w_test3_loraB');

        loraA.assign(tf.ones([outDim, rank]));
        loraB.assign(tf.ones([rank, inDim]));

        lora.detach();

        expect(weightStore.hasVariable('w_test3_loraA')).toBe(true);
        expect(weightStore.hasVariable('w_test3_loraB')).toBe(true);
        expect(weightStore.onWeightRead).toBeUndefined();

        lora.dispose();
    });

    it('dispose removes LoRA variables', async ({ expect }) => {
        await selectBackend('webgpu');

        const weightStore = new WeightStore();
        const outDim = 2;
        const inDim = 2;
        const rank = 2;
        const alpha = 6;

        const base = tf.variable(tf.zeros([outDim, inDim]), true, 'w');
        weightStore.addVariable('w', base);

        const lora = new LoRA('test4', weightStore, alpha, rank, ['w']);

        const loraA = weightStore.getRawVariable('w_test4_loraA');
        const loraB = weightStore.getRawVariable('w_test4_loraB');

        loraA.assign(tf.ones([outDim, rank]));
        loraB.assign(tf.ones([rank, inDim]));

        lora.dispose();

        expect(weightStore.hasVariable('w_test4_loraA')).toBe(false);
        expect(weightStore.hasVariable('w_test4_loraB')).toBe(false);
    });

    it('applies LoRA only through getVariable when attached', async ({ expect }) => {
        await selectBackend('webgpu');

        const weightStore = new WeightStore();
        const outDim = 2;
        const inDim = 3;
        const rank = 1;
        const alpha = 5; // scale = alpha / rank = 5

        // Use non-zero base so we can verify additive behavior
        const baseInit = tf.tensor2d(
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [outDim, inDim]
        );
        const base = tf.variable(baseInit, true, 'w');
        weightStore.addVariable('w', base);

        // Baseline read before attaching LoRA
        const before = weightStore.getVariable('w');
        const beforeData = Array.from(await before.data());
        expect(beforeData).toEqual([1, 2, 3, 4, 5, 6]);
        //before.dispose();

        const lora = new LoRA('test1', weightStore, alpha, rank, ['w']);
        lora.attach();

        // A @ B = [[3,4,5],[6,8,10]], then multiplied by scale(5)
        // delta = [[15,20,25],[30,40,50]]
        const loraA = weightStore.getRawVariable('w_test1_loraA');
        const loraB = weightStore.getRawVariable('w_test1_loraB');
        loraA.assign(tf.tensor2d([[1], [2]], [outDim, rank]));
        loraB.assign(tf.tensor2d([[3, 4, 5]], [rank, inDim]));

        // getVariable should include LoRA delta
        const adjusted = weightStore.getVariable('w');
        const adjustedData = Array.from(await adjusted.data());
        expect(adjustedData).toEqual([16, 22, 28, 34, 45, 56]);
        adjusted.dispose();

        // Raw variable should still be unmodified (LoRA is applied on read)
        const rawData = Array.from(await weightStore.getRawVariable('w').data());
        expect(rawData).toEqual([1, 2, 3, 4, 5, 6]);

        // After detach, getVariable should return base again
        lora.detach();
        const afterDetach = weightStore.getVariable('w');
        const afterDetachData = Array.from(await afterDetach.data());
        expect(afterDetachData).toEqual([1, 2, 3, 4, 5, 6]);
        afterDetach.dispose();

        lora.dispose();
        baseInit.dispose();
    });
});
