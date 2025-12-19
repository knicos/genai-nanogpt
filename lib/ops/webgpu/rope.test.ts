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
import { randomNormal } from '@tensorflow/tfjs-core';
import { rope } from '../rope';
import RoPECache from '@base/layers/RoPECache';
import { ropeGradConfig } from '../grads/rope';
import { isPackedTensor } from '@base/utilities/packed';

describe('Rope WebGPU', () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });
    it('is identical for packed and unpacked tensors', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([100, 2, 64, 64], 0, 1, 'float32');

        const packed = pack16(x);
        const unpacked = unpack16(packed);

        const cache = new RoPECache({
            biasInLayerNorm: false,
            vocabSize: 20,
            nEmbed: 128,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.0,
            blockSize: 64,
            mlpFactor: 4,
            useRope: true,
        });
        const packedRope = rope(packed, cache, 0);
        const unpackedRope = rope(unpacked, cache, 0);

        const repackedRope = unpack16(pack16(unpackedRope));
        const unpackedFromPacked = unpack16(packedRope);

        const unpackedData = await repackedRope.data();
        const unpackedFromPackedData = await unpackedFromPacked.data();

        const error = arraysClose(unpackedData, unpackedFromPackedData);
        expect(error).toBeLessThan(1e-3);
        expect(unpackedFromPackedData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
    });

    it('remains identical for non zero pastLen', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([100, 2, 64, 64], 0, 1, 'float32');

        const packed = pack16(x);
        const unpacked = unpack16(packed);

        const cache = new RoPECache({
            biasInLayerNorm: false,
            vocabSize: 20,
            nEmbed: 128,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.0,
            blockSize: 64,
            mlpFactor: 4,
            useRope: true,
        });
        const packedRope = rope(packed, cache, 10);
        const unpackedRope = rope(unpacked, cache, 10);

        const repackedRope = unpack16(pack16(unpackedRope));
        const unpackedFromPacked = unpack16(packedRope);

        const unpackedData = await repackedRope.data();
        const unpackedFromPackedData = await unpackedFromPacked.data();

        const error = arraysClose(unpackedData, unpackedFromPackedData);
        expect(error).toBeLessThan(1e-3);
    });

    it('has valid gradients', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([100, 2, 64, 64], 0, 1, 'float32');

        const cache = new RoPECache({
            biasInLayerNorm: false,
            vocabSize: 20,
            nEmbed: 128,
            nHead: 2,
            nLayer: 1,
            biasInLinear: false,
            dropout: 0.0,
            blockSize: 64,
            mlpFactor: 4,
            useRope: true,
        });

        const packedX = pack16(x);

        const gradX = ropeGradConfig.gradFunc(packedX, [cache.getSin()!, cache.getCos()!], {}).x();
        const gradXData = await gradX.data();

        expect(gradXData.length).toBe(packedX.size);
        expect(gradXData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
        expect(gradXData.some((v) => isNaN(v))).toBe(false);
        expect(gradXData.some((v) => !isFinite(v))).toBe(false);
        expect(gradX.dtype).toBe('int32');
        expect(isPackedTensor(gradX)).toBe(true);
    });
});
