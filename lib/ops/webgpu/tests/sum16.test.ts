import '@base/patches/engine';
import { afterAll, describe, it } from 'vitest';
import { create, globals } from 'webgpu';
import { pack16 } from '../../pack16';
import { unpack16 } from '../../unpack16';
import { arraysClose } from '@base/utilities/arrayClose';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';
import { randomNormal, sum } from '@tensorflow/tfjs-core';
import { sum16 } from '@base/ops/sum16';

describe('Sum 16-bit', { timeout: 10000 }, () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('should match unpacked sum without keepDims', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([100, 4, 64], 0, 1, 'float32');

        const packed = pack16(x);
        const unpackedX = unpack16(packed);

        const originalSum = unpack16(pack16(sum(unpackedX, [2], false)));
        const packedSum = sum16(packed, [2], false);
        const unpacked = unpack16(packedSum);

        const originalData = await originalSum.data();
        const unpackedData = await unpacked.data();

        expect(originalSum.shape).toEqual([100, 4]);
        expect(unpacked.shape).toEqual([100, 4]);

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);

        expect(unpackedData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
    });

    it('can sum over non last dimensions', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([100, 64], 0, 1, 'float32');

        const packed = pack16(x);
        const unpackedX = unpack16(packed);

        const originalSum = unpack16(pack16(sum(unpackedX, [0], false)));
        const packedSum = sum16(packed, [0], false);
        const unpacked = unpack16(packedSum);

        const originalData = await originalSum.data();
        const unpackedData = await unpacked.data();

        expect(originalSum.shape).toEqual([64]);
        expect(unpacked.shape).toEqual([64]);

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);

        expect(unpackedData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
    });
});
