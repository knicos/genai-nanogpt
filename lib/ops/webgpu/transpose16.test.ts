import '@base/patches/engine';
import { afterAll, describe, it } from 'vitest';
import { create, globals } from 'webgpu';
import { pack16 } from '../pack16';
import { unpack16 } from '../unpack16';
import { arraysClose } from '@base/utilities/arrayClose';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';
import { randomNormal, transpose } from '@tensorflow/tfjs-core';
import { transpose16 } from '../transpose16';

describe('Transpose 16-bit', () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });
    it('should match unpacked transpose', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([100, 4, 64], 0, 1, 'float32');

        const packed = pack16(x);
        const unpackedX = unpack16(packed);

        const originalTranspose = transpose(unpackedX, [1, 0, 2]);
        const packedTranspose = transpose16(packed, [1, 0, 2]);
        const unpacked = unpack16(packedTranspose);

        const originalData = await originalTranspose.data();
        const unpackedData = await unpacked.data();

        console.log('Upacked data:', unpackedData.slice(0, 10));

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);

        expect(unpackedData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
    });

    /* it('has valid gradients', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([50, 20], 0, 1, 'float32');
        const y = randomNormal([50, 20], 0, 1, 'float32');

        const packedX = pack16(x);
        const packedY = pack16(y);

        const gradX = softmax16GradConfig.gradFunc(packedY, [packedX], { dim: 1 }).logits();
        const gradXData = await gradX.data();

        expect(gradXData.length).toBe(packedX.size);
        expect(gradXData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
        expect(gradXData.some((v) => isNaN(v))).toBe(false);
        expect(gradXData.some((v) => !isFinite(v))).toBe(false);
        expect(gradX.dtype).toBe('int32');
        expect(isPackedTensor(gradX)).toBe(true);
    });*/
});
