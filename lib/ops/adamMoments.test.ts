import '@base/patches/engine';
import { AdamOptimizer, randomNormal, zeros } from '@tensorflow/tfjs';
import { describe, it } from 'vitest';
import { adamMoments } from './adamMoments';
import { arraysClose } from '@base/utilities/arrayClose';

describe('adamMoments CPU kernel', () => {
    it('should compute new moments correctly', async ({ expect }) => {
        const realOptimiser = new AdamOptimizer(0.001, 0.99, 0.95, 1e-8);

        const moments = zeros([16, 16, 2]); // shape [height, width, 2] for m1 and m2
        const gradient = randomNormal([16, 16]);
        const variable = randomNormal([16, 16]).variable(true, 'var');

        realOptimiser.applyGradients([{ name: 'var', tensor: gradient }]);

        const newMoments = adamMoments(moments, gradient, 0.99, 0.95, 1);

        const updatedM1 = newMoments.slice([0, 0, 0], [-1, -1, 1]).squeeze([-1]);
        const updatedM2 = newMoments.slice([0, 0, 1], [-1, -1, 1]).squeeze([-1]);

        const realM1 = (await realOptimiser.getWeights()).find((w) => w.name === 'var/m')!.tensor;
        const realM2 = (await realOptimiser.getWeights()).find((w) => w.name === 'var/v')!.tensor;

        const m1RealArray = await realM1.array();
        const m2RealArray = await realM2.array();
        const m1TestArray = await updatedM1.array();
        const m2TestArray = await updatedM2.array();

        const close1 = arraysClose(m1RealArray, m1TestArray) <= 1e-6;
        const close2 = arraysClose(m2RealArray, m2TestArray) <= 1e-6;

        expect(close1).toBe(true);
        expect(close2).toBe(true);

        variable.dispose();
        gradient.dispose();
        moments.dispose();
        newMoments.dispose();
        realM1.dispose();
        realM2.dispose();
    });
});
