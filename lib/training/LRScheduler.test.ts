import { describe, it } from 'vitest';
import LRScheduler from './LRScheduler';

describe('LR Scheduler', () => {
    it('linearly warms up learning rate', ({ expect }) => {
        const scheduler = new LRScheduler(0.01, {
            warmupSteps: 4,
            decayEpochs: 2,
            minLearningRate: 0.001,
            epochSteps: 10,
        });

        expect(scheduler.getNextLR()).toBeCloseTo(0.0025, 8);
        expect(scheduler.getNextLR()).toBeCloseTo(0.005, 8);
        expect(scheduler.getNextLR()).toBeCloseTo(0.0075, 8);
        expect(scheduler.getNextLR()).toBeCloseTo(0.01, 8);
    });

    it('cosine decays and clamps to min learning rate', ({ expect }) => {
        const scheduler = new LRScheduler(0.01, {
            warmupSteps: 0,
            decayEpochs: 1,
            minLearningRate: 0.001,
            epochSteps: 4,
        });

        const lr0 = scheduler.getNextLR();
        const lr1 = scheduler.getNextLR();
        const lr2 = scheduler.getNextLR();
        const lr3 = scheduler.getNextLR();
        const lr4 = scheduler.getNextLR();

        expect(lr0).toBeCloseTo(0.01, 8);
        expect(lr1).toBeLessThan(lr0);
        expect(lr2).toBeLessThan(lr1);
        expect(lr3).toBeLessThan(lr2);
        expect(lr3).toBeGreaterThan(0.001);
        expect(lr4).toBeCloseTo(0.001, 8);
        expect(scheduler.lr).toBeCloseTo(0.001, 8);
    });

    it('updates scheduler config and starting learning rate', ({ expect }) => {
        const scheduler = new LRScheduler(0.02, {
            warmupSteps: 0,
            decayEpochs: 2,
            minLearningRate: 0.002,
            epochSteps: 10,
        });

        scheduler.updateConfig({ warmupSteps: 2, minLearningRate: 0.005 }, 0.03);

        expect(scheduler.getNextLR()).toBeCloseTo(0.015, 8);
        expect(scheduler.getNextLR()).toBeCloseTo(0.03, 8);

        const serialised = scheduler.serializeConfig();
        expect(serialised.warmupSteps).toBe(2);
        expect(serialised.minLearningRate).toBe(0.005);
        expect(serialised.step).toBe(2);
    });
});
