import '@base/patches/engine';
import { afterAll, describe, it } from 'vitest';
import { create, globals } from 'webgpu';
import { unpack16 } from '../../unpack16';
import { arraysClose } from '@base/utilities/arrayClose';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';
import { randomNormal } from '@tensorflow/tfjs-core';
import { qkv } from '../../qkv';
import { isPackedTensor } from '@base/utilities/packed';
import { pack16 } from '../../pack16';
import { qkvGrad } from '@base/ops/grads/qkv';

describe('QKV 16-bit', () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    it('should match unpacked QKV', async ({ expect }) => {
        await selectBackend('webgpu');
        const kernel = randomNormal([64, 32 * 2 * 3]);
        const input = randomNormal([1, 128, 64]);

        const packedInput = pack16(input);
        const packedKernel = pack16(kernel);
        const [packedQ, packedK, packedV] = qkv(packedInput, packedKernel, 2);
        const unpackedQ = unpack16(packedQ);
        const unpackedK = unpack16(packedK);
        const unpackedV = unpack16(packedV);

        const [originalQ, originalK, originalV] = qkv(unpack16(packedInput), unpack16(packedKernel), 2);

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

    it('supports a larger batch dimension', async ({ expect }) => {
        await selectBackend('webgpu');
        const kernel = randomNormal([64, 32 * 2 * 3]);
        const input = randomNormal([32, 128, 64]);

        const packedInput = pack16(input);
        const packedKernel = pack16(kernel);
        const [packedQ, packedK, packedV] = qkv(packedInput, packedKernel, 2);
        const unpackedQ = unpack16(packedQ);
        const unpackedK = unpack16(packedK);
        const unpackedV = unpack16(packedV);

        const [originalQ, originalK, originalV] = qkv(unpack16(packedInput), unpack16(packedKernel), 2);

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

    it('should produce similar gradients for each precision', async ({ expect }) => {
        await selectBackend('webgpu');
        const kernel = randomNormal([64, 32 * 2 * 3]);
        const input = randomNormal([1, 128, 64]);

        const packedInput = pack16(input);
        const packedKernel = pack16(kernel);

        const q = randomNormal([1, 2, 128, 32]);
        const k = randomNormal([1, 2, 128, 32]);
        const v = randomNormal([1, 2, 128, 32]);

        const packedQ = pack16(q);
        const packedK = pack16(k);
        const packedV = pack16(v);

        const gradPacked = qkvGrad([packedQ, packedK, packedV], packedInput, packedKernel);
        const gradPackedX = gradPacked.x();
        const gradPackedKernel = gradPacked.kernel();

        const gradUnpacked = qkvGrad(
            [unpack16(packedQ), unpack16(packedK), unpack16(packedV)],
            unpack16(packedInput),
            unpack16(packedKernel)
        );
        const gradUnpackedX = pack16(gradUnpacked.x());
        const gradUnpackedKernel = pack16(gradUnpacked.kernel());

        const gradPackedXData = await unpack16(gradPackedX).data();
        const gradUnpackedXData = await unpack16(gradUnpackedX).data();
        const gradPackedKernelData = await unpack16(gradPackedKernel).data();
        const gradUnpackedKernelData = await unpack16(gradUnpackedKernel).data();

        const errorX = arraysClose(gradPackedXData, gradUnpackedXData);
        expect(errorX).toBeLessThan(1e-3);
        const errorKernel = arraysClose(gradPackedKernelData, gradUnpackedKernelData);
        expect(errorKernel).toBeLessThan(1e-3);
    });
});
