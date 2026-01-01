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
import { pad, randomNormal } from '@tensorflow/tfjs-core';
import { packGradConfig } from '../../grads/pack16';
import { unpackGradConfig } from '../../grads/unpack16';
import { isPackedTensor } from '@base/utilities/packed';

describe('Pack and Unpack 16-bit floats', { timeout: 30000 }, () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });
    it('should pack and unpack correctly', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([100, 64], 0, 1, 'float32');

        const packed = pack16(x);
        const unpacked = unpack16(packed);

        const xData = await x.data();
        const unpackedData = await unpacked.data();

        const error = arraysClose(xData, unpackedData);
        expect(error).toBeLessThan(2e-3);
    });

    it('should pack and unpack with padding', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([128, 68], 0, 1, 'float32');

        const packed = pack16(x, undefined, 32);
        const unpacked = unpack16(packed);

        const paddedX = pad(x, [
            [0, 0],
            [0, 28],
        ]);
        const xData = await paddedX.data();
        const unpackedData = await unpacked.data();

        const error = arraysClose(xData, unpackedData);
        expect(error).toBeLessThan(2e-3);

        expect(unpacked.shape).toEqual([128, 64 + 32]);
    });

    it('should pack and unpack with padding on outer', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([120, 64], 0, 1, 'float32');

        const packed = pack16(x, undefined, 32);
        const unpacked = unpack16(packed);

        const paddedX = pad(x, [
            [0, 8],
            [0, 0],
        ]);
        const xData = await paddedX.data();
        const unpackedData = await unpacked.data();

        const error = arraysClose(xData, unpackedData);
        expect(error).toBeLessThan(2e-3);

        expect(unpacked.shape).toEqual([128, 64]);
    });

    it('has valid gradients for pack16', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([50, 20], 0, 1, 'float32');

        const packed = pack16(x);

        const gradX = packGradConfig.gradFunc(packed, [], {}).x();
        const gradXData = await gradX.data();

        expect(gradXData.length).toBe(x.size);
        expect(gradXData.every((v) => v === 0)).toBe(false);
        expect(gradXData.some((v) => isNaN(v))).toBe(false);
        expect(gradXData.some((v) => !isFinite(v))).toBe(false);
        expect(gradX.dtype).toBe('float32');
    });

    it('has valid gradients for unpack16', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([50, 20], 0, 1, 'float32');

        const gradX = unpackGradConfig.gradFunc(x, [], {}).x();
        const unpackedGradX = unpack16(gradX);
        const unpackedGradXData = await unpackedGradX.data();
        const gradXData = await gradX.data();

        expect(gradXData.length).toBe(x.size / 2);
        expect(unpackedGradXData.every((v) => v === 0)).toBe(false);
        expect(unpackedGradXData.some((v) => isNaN(v))).toBe(false);
        expect(unpackedGradXData.some((v) => !isFinite(v))).toBe(false);
        expect(gradX.dtype).toBe('int32');
        expect(isPackedTensor(gradX)).toBe(true);
    });
});
