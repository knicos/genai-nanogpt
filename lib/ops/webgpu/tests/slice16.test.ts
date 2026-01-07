import { afterAll, describe, it } from 'vitest';
import { create, globals } from 'webgpu';
import { pack16 } from '../../pack16';
import { unpack16 } from '../../unpack16';
import { arraysClose } from '@base/utilities/arrayClose';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';
import { randomNormal, slice } from '@tensorflow/tfjs-core';
import { slice16 } from '@base/ops/slice16';

describe('Slice 16-bit', { timeout: 10000 }, () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('should match unpacked slice', async ({ expect }) => {
        await selectBackend('webgpu');
        const a = randomNormal([100, 4, 64], 0, 1, 'float32');

        const packedA = pack16(a);
        const unpackedA = unpack16(packedA);

        const originalSlice = slice(unpackedA, [0, 1, 10], [100, 2, 20]);
        // Good to note that the last dimension is halved when slicing packed tensors
        const packedSlice = slice16(packedA, [0, 1, 10 / 2], [100, 2, 20 / 2]);
        const unpacked = unpack16(packedSlice);

        const originalData = await originalSlice.data();
        const unpackedData = await unpacked.data();

        expect(originalSlice.shape).toEqual([100, 2, 20]);
        expect(unpacked.shape).toEqual([100, 2, 20]);

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);
    });
});
