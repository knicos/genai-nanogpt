import '@base/patches/engine';
import { afterAll, describe, it } from 'vitest';
import { create, globals } from 'webgpu';
import { unpack16 } from '../unpack16';
import { arraysClose } from '@base/utilities/arrayClose';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';
import { randomNormal } from '@tensorflow/tfjs-core';
import { qkv } from '../qkv';
import { isPackedTensor } from '@base/utilities/packed';
import { pack16 } from '../pack16';

describe('QKV 16-bit', () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });
    it('should match unpacked QKV', async ({ expect }) => {
        await selectBackend('webgpu');
        const kernel = randomNormal([16, 48]);
        const input = randomNormal([1, 4, 16]);

        const [packedQ, packedK, packedV] = qkv(input, kernel, 4, true);
        const unpackedQ = unpack16(packedQ);
        const unpackedK = unpack16(packedK);
        const unpackedV = unpack16(packedV);

        const [originalQ, originalK, originalV] = qkv(input, kernel, 4, false);

        const repackedQ = unpack16(pack16(originalQ));
        const repackedK = unpack16(pack16(originalK));
        const repackedV = unpack16(pack16(originalV));

        const originalQData = await repackedQ.data();
        const unpackedQData = await unpackedQ.data();
        const originalKData = await repackedK.data();
        const unpackedKData = await unpackedK.data();
        const originalVData = await repackedV.data();
        const unpackedVData = await unpackedV.data();

        const errorQ = arraysClose(originalQData, unpackedQData);
        expect(errorQ).toBeLessThan(1e-3);
        const errorK = arraysClose(originalKData, unpackedKData);
        expect(errorK).toBeLessThan(1e-3);
        const errorV = arraysClose(originalVData, unpackedVData);
        expect(errorV).toBeLessThan(1e-3);

        expect(packedQ.dtype).toBe('int32');
        expect(packedK.dtype).toBe('int32');
        expect(packedV.dtype).toBe('int32');
        expect(isPackedTensor(packedQ)).toBe(true);
        expect(isPackedTensor(packedK)).toBe(true);
        expect(isPackedTensor(packedV)).toBe(true);

        expect(unpackedQData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
        expect(unpackedKData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
        expect(unpackedVData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
    });

    /*it('has valid gradients', async ({ expect }) => {
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
