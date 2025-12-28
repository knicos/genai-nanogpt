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
import { randomNormal, transpose } from '@tensorflow/tfjs-core';
import { transpose16 } from '../../transpose16';

describe('Transpose 16-bit', () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('should transpose first of rank 3', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([100, 4, 64], 0, 1, 'float32');

        const packed = pack16(x);
        const unpackedX = unpack16(packed);

        const originalTranspose = transpose(unpackedX, [1, 0, 2]);
        const packedTranspose = transpose16(packed, [1, 0, 2]);
        const unpacked = unpack16(packedTranspose);

        const originalData = await originalTranspose.data();
        const unpackedData = await unpacked.data();

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);

        expect(unpackedData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
    });

    it('should transpose last of rank 3', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([100, 4, 64], 0, 1, 'float32');

        const packed = pack16(x);
        const unpackedX = unpack16(packed);

        const originalTranspose = transpose(unpackedX, [0, 2, 1]);
        const packedTranspose = transpose16(packed, [0, 2, 1]);
        const unpacked = unpack16(packedTranspose);

        const originalData = await originalTranspose.data();
        const unpackedData = await unpacked.data();

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);

        expect(unpackedData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
    });

    it('should transpose middle of rank 4', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([100, 4, 64, 32], 0, 1, 'float32');

        const packed = pack16(x);
        const unpackedX = unpack16(packed);

        const originalTranspose = transpose(unpackedX, [0, 2, 1, 3]);
        const packedTranspose = transpose16(packed, [0, 2, 1, 3]);
        const unpacked = unpack16(packedTranspose);

        const originalData = await originalTranspose.data();
        const unpackedData = await unpacked.data();

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);

        expect(unpackedData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
    });
});
