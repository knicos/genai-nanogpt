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
import { add, randomNormal } from '@tensorflow/tfjs-core';
import { add16 } from '@base/ops/add16';

describe('Add 16-bit', { timeout: 10000 }, () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('should match unpacked add', async ({ expect }) => {
        await selectBackend('webgpu');
        const a = randomNormal([100, 4, 64], 0, 1, 'float32');
        const b = randomNormal([100, 4, 64], 0, 1, 'float32');

        const packedA = pack16(a);
        const packedB = pack16(b);
        const unpackedA = unpack16(packedA);
        const unpackedB = unpack16(packedB);

        const originalAdd = unpack16(pack16(add(unpackedA, unpackedB)));
        const packedAdd = add16(packedA, packedB);
        const unpacked = unpack16(packedAdd);

        const originalData = await originalAdd.data();
        const unpackedData = await unpacked.data();

        expect(originalAdd.shape).toEqual([100, 4, 64]);
        expect(unpacked.shape).toEqual([100, 4, 64]);

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);
    });
});
