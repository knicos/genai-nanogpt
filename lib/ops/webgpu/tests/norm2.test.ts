import { afterAll, describe, it } from 'vitest';
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';
import { engine, ones, Tensor, zeros } from '@tensorflow/tfjs';
import '../norm2';
import { clipScale } from '@base/ops/globalNorm';

function int32BitsToFloat32(value: number): number {
    const intView = new Int32Array(1);
    const floatView = new Float32Array(intView.buffer);
    intView[0] = value;
    return floatView[0];
}

describe('Norm squared', { timeout: 10000 }, () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('should produce a single scalar', async ({ expect }) => {
        await selectBackend('webgpu');
        const a = ones([100, 4, 64], 'float32');
        const output = zeros([1], 'int32');

        const norm2 = engine().runKernel('Norm2', { x: a, output }, { invLossScaling: 1.0, index: 0 }) as Tensor;

        const data = await norm2.data();
        const norm = int32BitsToFloat32(data[0]);

        expect(data).toHaveLength(1);
        expect(norm).toBeCloseTo(100 * 4 * 64, 3);
    });

    it('should work with loss scaling', async ({ expect }) => {
        await selectBackend('webgpu');
        const a = ones([100, 4, 64], 'float32');
        const output = zeros([1], 'int32');

        const norm2 = engine().runKernel('Norm2', { x: a, output }, { invLossScaling: 0.5, index: 0 }) as Tensor;

        const data = await norm2.data();
        const norm = int32BitsToFloat32(data[0]);

        expect(data).toHaveLength(1);
        expect(norm).toBeCloseTo(100 * 4 * 64 * 0.5 ** 2, 3);
    });

    it('can pack multiple results', async ({ expect }) => {
        await selectBackend('webgpu');
        const a = ones([100, 4, 64], 'float32');
        const b = ones([100, 4, 32], 'float32');
        const output = zeros([2], 'int32');

        engine().runKernel('Norm2', { x: a, output }, { invLossScaling: 1.0, index: 0 });
        engine().runKernel('Norm2', { x: b, output }, { invLossScaling: 1.0, index: 1 });

        const data = await output.data();
        const normA = int32BitsToFloat32(data[0]);
        const normB = int32BitsToFloat32(data[1]);

        expect(data).toHaveLength(2);
        expect(normA).toBeCloseTo(100 * 4 * 64, 3);
        expect(normB).toBeCloseTo(100 * 4 * 32, 3);
    });

    it('results in a clipScale', async ({ expect }) => {
        await selectBackend('webgpu');
        const a = ones([100, 4, 64], 'float32');
        const b = ones([100, 4, 64], 'float32');

        const cs = clipScale([a, b], 0.5, 1.0);
        const data = await cs.data();

        const expectedNorm = Math.sqrt(100 * 4 * 64 * 2 * 0.5 ** 2);
        const expectedScale = 0.5 / Math.max(expectedNorm, 1.0);

        expect(data).toHaveLength(2);
        expect(data[0]).toBeCloseTo(expectedScale, 3);
    });

    it('handles very large gradients without NaNs', async ({ expect }) => {
        await selectBackend('webgpu');
        const a = ones([100, 4, 64], 'float32').mul(1e12);
        const b = ones([100, 4, 64], 'float32').mul(1e12);
        const output = zeros([2], 'int32');

        engine().runKernel('Norm2', { x: a, output }, { invLossScaling: 1.0, index: 0 });
        engine().runKernel('Norm2', { x: b, output }, { invLossScaling: 1.0, index: 1 });

        const normDataBits = await output.data();
        const normA = int32BitsToFloat32(normDataBits[0]);
        const normB = int32BitsToFloat32(normDataBits[1]);

        expect(Number.isFinite(normA)).toBe(true);
        expect(Number.isFinite(normB)).toBe(true);
        expect(normA).toBeGreaterThan(0);
        expect(normB).toBeGreaterThan(0);

        const cs = clipScale([a, b], 1.0, 1.0);
        const clipData = await cs.data();

        expect(Number.isFinite(clipData[0])).toBe(true);
        expect(Number.isFinite(clipData[1])).toBe(true);
        expect(Number.isNaN(clipData[0])).toBe(false);
        expect(Number.isNaN(clipData[1])).toBe(false);
    });
});
