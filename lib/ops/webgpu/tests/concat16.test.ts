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
import { randomNormal } from '@tensorflow/tfjs-core';
import { concat16 } from '@base/ops/concat16';

describe('Concat 16-bit', () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('should match unpacked concat', async ({ expect }) => {
        await selectBackend('webgpu');
        const a = randomNormal([100, 4, 64], 0, 1, 'float32');
        const b = randomNormal([100, 4, 64], 0, 1, 'float32');

        const packedA = pack16(a);
        const packedB = pack16(b);
        const unpackedA = unpack16(packedA);
        const unpackedB = unpack16(packedB);

        const originalConcat = unpack16(pack16(concat16([unpackedA, unpackedB], 2)));
        const packedConcat = concat16([packedA, packedB], 2);
        const unpacked = unpack16(packedConcat);

        const originalData = await originalConcat.data();
        const unpackedData = await unpacked.data();

        expect(originalConcat.shape).toEqual([100, 4, 128]);
        expect(unpacked.shape).toEqual([100, 4, 128]);

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);
    });

    it('should concat on second dimension', async ({ expect }) => {
        await selectBackend('webgpu');
        const a = randomNormal([100, 4, 64], 0, 1, 'float32');
        const b = randomNormal([100, 4, 64], 0, 1, 'float32');

        const packedA = pack16(a);
        const packedB = pack16(b);
        const unpackedA = unpack16(packedA);
        const unpackedB = unpack16(packedB);

        const originalConcat = unpack16(pack16(concat16([unpackedA, unpackedB], 1)));
        const packedConcat = concat16([packedA, packedB], 1);
        const unpacked = unpack16(packedConcat);

        const originalData = await originalConcat.data();
        const unpackedData = await unpacked.data();

        expect(originalConcat.shape).toEqual([100, 8, 64]);
        expect(unpacked.shape).toEqual([100, 8, 64]);

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);
    });

    it('should concat on first dimension', async ({ expect }) => {
        await selectBackend('webgpu');
        const a = randomNormal([100, 4, 64], 0, 1, 'float32');
        const b = randomNormal([100, 4, 64], 0, 1, 'float32');

        const packedA = pack16(a);
        const packedB = pack16(b);
        const unpackedA = unpack16(packedA);
        const unpackedB = unpack16(packedB);

        const originalConcat = unpack16(pack16(concat16([unpackedA, unpackedB], 0)));
        const packedConcat = concat16([packedA, packedB], 0);
        const unpacked = unpack16(packedConcat);

        const originalData = await originalConcat.data();
        const unpackedData = await unpacked.data();

        expect(originalConcat.shape).toEqual([200, 4, 64]);
        expect(unpacked.shape).toEqual([200, 4, 64]);

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-3);
    });
});
