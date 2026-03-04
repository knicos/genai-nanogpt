import { afterAll, describe, it } from 'vitest';
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';
import { grad, ones, scalar, sum, Tensor } from '@tensorflow/tfjs';
import { dropout16 } from '../../dropout16';
import { pack16 } from '@base/ops/pack16';
import { unpack16 } from '@base/ops/unpack16';

describe('Dropout16', { timeout: 10000 }, () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('should mask some outputs', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = ones([100, 4, 64], 'float32');
        const packedX = pack16(x);

        const output = dropout16(packedX, 0.5);
        const unpackedOutput = unpack16(output);

        const data = await unpackedOutput.data();

        const numZeros = data.filter((v) => v === 0).length;
        const numNonZeros = data.length - numZeros;

        // With a dropout rate of 0.5, we expect about half the values to be zero
        expect(numZeros).toBeGreaterThan(0.4 * data.length);
        expect(numNonZeros).toBeGreaterThan(0.4 * data.length);
    });

    it('gradient should match forward dropout mask (fixed seed)', async ({ expect }) => {
        await selectBackend('webgpu');

        const shape: [number, number, number] = [32, 4, 64];
        const rate = 0.5;
        const seed = Math.random();

        const x = ones(shape, 'float32');
        const packedX = pack16(x);

        // Forward (deterministic mask via seed)
        const yPacked = dropout16(packedX, rate, seed);
        const y = unpack16(yPacked);

        // d/dx sum(dropout16(x)) => dy is all ones; gradient should use same mask/scale
        const gradFn = grad((inp: Tensor) => sum(unpack16(dropout16(inp, rate, seed))));
        const dxPacked = gradFn(packedX, scalar(1));
        const dx = unpack16(dxPacked);

        const [yData, dxData] = await Promise.all([y.data(), dx.data()]);

        let mismatchedMask = 0;
        let mismatchedKeptValues = 0;

        for (let i = 0; i < yData.length; i++) {
            const yZero = yData[i] === 0;
            const dxZero = dxData[i] === 0;

            if (yZero !== dxZero) {
                mismatchedMask++;
            } else if (!yZero && Math.abs(yData[i] - dxData[i]) > 1e-3) {
                mismatchedKeptValues++;
            }
        }

        expect(mismatchedMask).toBe(0);
        expect(mismatchedKeptValues).toBe(0);
    });
});
