import { afterAll, beforeEach, describe, it } from 'vitest';
import { array } from '@tensorflow/tfjs-data';
import { create, globals } from 'webgpu';
import { tensor } from '@tensorflow/tfjs-core';
import createModelInstance from '@base/models/factory';
import CharTokeniser from '@base/tokeniser/CharTokeniser';
import BasicTrainer from './BasicTrainer';
import { AdamWOptimizer } from './AdamW';
import { arraysClose } from '@base/utilities/arrayClose';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';

describe('Basic Trainer', () => {
    beforeEach(async () => {
        await selectBackend('webgpu');
    });

    it('should train a simple model for one step and update weights', async ({ expect }) => {
        const model = createModelInstance({
            modelType: 'GenAI_NanoGPT_v1',
            vocabSize: 32,
            blockSize: 4,
            nLayer: 1,
            nHead: 1,
            nEmbed: 4,
            mlpFactor: 2,
            useRope: false,
        });

        const tokeniser = new CharTokeniser(32);
        const optimizer = new AdamWOptimizer({
            learningRate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weightDecay: 0.0,
            warmupSteps: 0,
            decayEpochs: 1,
            minLearningRate: 1e-3,
            epochSteps: 100,
            lossScaling: 1,
        });

        const trainer = new BasicTrainer(model, tokeniser, undefined, optimizer);

        await (
            trainer as unknown as {
                dummyPass: () => Promise<void>;
            }
        ).dummyPass();

        expect(model.weightStore.variables.length).toBeGreaterThan(0);

        const beforeWeights = new Map<string, number[]>();
        for (const variable of model.weightStore.variables) {
            const name = variable.name;
            beforeWeights.set(name, Array.from((await variable.array()) as number[]));
        }

        const xs = tensor([[1, 2, 3, 4]], [1, 4], 'int32');
        const ys = tensor([[2, 3, 4, 5]], [1, 4], 'int32');

        const dataset = array([{ xs, ys }]);
        await trainer.trainOnDataset(dataset, {
            batchSize: 1,
            maxEpochs: 1,
            epochSteps: 1,
            logInterval: 1,
        });

        expect(model.trainingState).toBeTruthy();
        expect(model.weightStore.variables.length).toBe(beforeWeights.size);

        let changed = false;
        for (const variable of model.weightStore.variables) {
            const name = variable.name;
            const before = beforeWeights.get(name);
            if (!before) continue;
            const after = Array.from((await variable.array()) as number[]);
            if (arraysClose(before, after) > 1e-8) {
                changed = true;
                break;
            }
        }

        expect(changed).toBe(true);

        trainer.dispose();
        model.dispose();
    });

    it('should train a v2 model for one step and update weights', async ({ expect }) => {
        const model = createModelInstance({
            modelType: 'GenAI_NanoGPT_v2',
            vocabSize: 32,
            blockSize: 4,
            nLayer: 1,
            nHead: 1,
            nEmbed: 4,
            mlpFactor: 2,
        });

        const tokeniser = new CharTokeniser(32);
        const optimizer = new AdamWOptimizer({
            learningRate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weightDecay: 0.0,
            warmupSteps: 0,
            decayEpochs: 1,
            minLearningRate: 1e-3,
            epochSteps: 100,
            lossScaling: 1,
        });

        const trainer = new BasicTrainer(model, tokeniser, undefined, optimizer);

        await (
            trainer as unknown as {
                dummyPass: () => Promise<void>;
            }
        ).dummyPass();

        expect(model.weightStore.variables.length).toBeGreaterThan(0);

        const beforeWeights = new Map<string, number[]>();
        for (const variable of model.weightStore.variables) {
            const name = variable.name;
            beforeWeights.set(name, Array.from((await variable.array()) as number[]));
        }

        const xs = tensor([[1, 2, 3, 4]], [1, 4], 'int32');
        const ys = tensor([[2, 3, 4, 5]], [1, 4], 'int32');

        const dataset = array([{ xs, ys }]);
        await trainer.trainOnDataset(dataset, {
            batchSize: 1,
            maxEpochs: 1,
            epochSteps: 1,
            logInterval: 1,
        });

        expect(model.trainingState).toBeTruthy();
        expect(model.weightStore.variables.length).toBe(beforeWeights.size);

        let changed = false;
        for (const variable of model.weightStore.variables) {
            const name = variable.name;
            const before = beforeWeights.get(name);
            if (!before) continue;
            const after = Array.from((await variable.array()) as number[]);
            if (arraysClose(before, after) > 1e-8) {
                changed = true;
                break;
            }
        }

        expect(changed).toBe(true);

        trainer.dispose();
        model.dispose();
    });

    /*it('should run one step via stepDataset and update weights', async ({ expect }) => {
        const model = createModelInstance({
            modelType: 'GenAI_NanoGPT_v1',
            vocabSize: 32,
            blockSize: 4,
            nLayer: 1,
            nHead: 1,
            nEmbed: 4,
            mlpFactor: 2,
            useRope: false,
        });

        const tokeniser = new CharTokeniser(32);
        const optimizer = new AdamWOptimizer({
            learningRate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weightDecay: 0.0,
            warmupSteps: 0,
            decayEpochs: 1,
            minLearningRate: 1e-3,
            epochSteps: 100,
            lossScaling: 1,
        });

        const trainer = new BasicTrainer(model, tokeniser, undefined, optimizer);

        await (
            trainer as unknown as {
                dummyPass: () => Promise<void>;
            }
        ).dummyPass();

        expect(model.weightStore.variables.length).toBeGreaterThan(0);

        const beforeWeights = new Map<string, number[]>();
        for (const variable of model.weightStore.variables) {
            const name = variable.name;
            beforeWeights.set(name, Array.from((await variable.array()) as number[]));
        }

        const xs = tensor([[1, 2, 3, 4]], [1, 4], 'int32');
        const ys = tensor([[2, 3, 4, 5]], [1, 4], 'int32');

        const dataset = array([{ xs, ys }]);
        await expect(
            trainer.stepDataset(dataset, {
                batchSize: 1,
                epochSteps: 1,
                logInterval: 1,
            })
        ).rejects.toThrow('No log returned before training stopped.');

        expect(model.trainingState).toBeTruthy();
        expect(model.weightStore.variables.length).toBe(beforeWeights.size);

        let changed = false;
        for (const variable of model.weightStore.variables) {
            const name = variable.name;
            const before = beforeWeights.get(name);
            if (!before) continue;
            const after = Array.from((await variable.array()) as number[]);
            if (arraysClose(before, after) > 1e-8) {
                changed = true;
                break;
            }
        }

        expect(changed).toBe(true);

        trainer.dispose();
        model.dispose();
    });*/

    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });
});
