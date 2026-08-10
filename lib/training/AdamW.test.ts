import { afterAll, beforeEach, describe, it } from 'vitest';
import { create, globals } from 'webgpu';
import { slice, squeeze, tensor, zeros } from '@tensorflow/tfjs-core';
import { AdamWOptimizer } from './AdamW';
import { load_safetensors } from '@base/utilities/safetensors';
import { arraysClose } from '@base/utilities/arrayClose';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';

describe('AdamW Optimizer', () => {
    beforeEach(async () => {
        await selectBackend('webgpu');
    });

    it('applies gradients and updates variables', async ({ expect }) => {
        const optimizer = new AdamWOptimizer({
            learningRate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            weightDecay: 0,
            lossScaling: 1,
            warmupSteps: 0,
            decayEpochs: 1,
            minLearningRate: 0.1,
            epochSteps: 1,
        });

        const variable = tensor([1, -1], [2], 'float32').variable(true, 'w_apply');
        const gradient = tensor([0.5, -0.5], [2], 'float32');
        const before = await variable.array();

        const scaling = optimizer.applyGradients([{ name: 'w_apply', tensor: gradient }]);
        const scalingData = await scaling.data();
        const after = await variable.array();
        const config = optimizer.serializeConfig();

        expect(Array.from(scalingData)).toEqual([1]);
        expect(after).not.toEqual(before);
        expect(config.accBeta1).toBeCloseTo(0.81, 6);
        expect(config.accBeta2).toBeCloseTo(0.998001, 6);

        scaling.dispose();
        gradient.dispose();
        variable.dispose();
        optimizer.dispose();
    });

    it('updates first and second moments based on gradients', async ({ expect }) => {
        const optimizer = new AdamWOptimizer({
            learningRate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            weightDecay: 0,
            lossScaling: 1,
            warmupSteps: 0,
            decayEpochs: 1,
            minLearningRate: 0.01,
            epochSteps: 1,
        });

        const variable = zeros([2]).variable(true, 'w_moments');
        const gradient = tensor([1, -2], [2], 'float32');

        const scaling = optimizer.applyGradients([{ name: 'w_moments', tensor: gradient }]);
        scaling.dispose();

        const momentsBin = await optimizer.saveMoments();
        const moments = await load_safetensors(momentsBin);
        const momentTensor = moments['w_moments/m'];

        expect(momentTensor).toBeDefined();
        expect(momentTensor.shape).toEqual([2, 2]);

        const mTensor = squeeze(slice(momentTensor, [0, 0], [2, 1]), [-1]);
        const vTensor = squeeze(slice(momentTensor, [0, 1], [2, 1]), [-1]);
        const mArray = await mTensor.array();
        const vArray = await vTensor.array();

        expect(arraysClose(mArray, [0.1, -0.2])).toBeLessThanOrEqual(1e-6);
        expect(arraysClose(vArray, [0.001, 0.004])).toBeLessThanOrEqual(1e-6);

        mTensor.dispose();
        vTensor.dispose();
        gradient.dispose();
        variable.dispose();
        Object.values(moments).forEach((t) => t.dispose());
        optimizer.dispose();
    });

    it('saves and loads optimizer moments', async ({ expect }) => {
        const config = {
            learningRate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            weightDecay: 0,
            lossScaling: 1,
            warmupSteps: 0,
            decayEpochs: 1,
            minLearningRate: 0.01,
            epochSteps: 1,
        };

        const optimizerA = new AdamWOptimizer(config);
        const variable = tensor([0.3, -0.1], [2], 'float32').variable(true, 'w_state');
        const gradient = tensor([0.25, -0.5], [2], 'float32');

        const scaling = optimizerA.applyGradients([{ name: 'w_state', tensor: gradient }]);
        scaling.dispose();

        const saved = await optimizerA.saveMoments();

        const optimizerB = new AdamWOptimizer(config);
        await optimizerB.loadMoments(saved);
        const reSaved = await optimizerB.saveMoments();

        const momentsA = await load_safetensors(saved);
        const momentsB = await load_safetensors(reSaved);

        expect(Object.keys(momentsB)).toEqual(Object.keys(momentsA));

        const key = Object.keys(momentsA)[0];
        const aArray = await momentsA[key].array();
        const bArray = await momentsB[key].array();
        expect(arraysClose(aArray, bArray)).toBeLessThanOrEqual(1e-6);

        gradient.dispose();
        variable.dispose();
        Object.values(momentsA).forEach((t) => t.dispose());
        Object.values(momentsB).forEach((t) => t.dispose());
        optimizerA.dispose();
        optimizerB.dispose();
    });

    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });
});
