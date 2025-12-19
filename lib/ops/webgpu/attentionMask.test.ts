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
import { matMul, randomNormal } from '@tensorflow/tfjs-core';
import { attentionMask } from '../attentionMask';

describe('Attention mask 16 bit', () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });
    it('should produce a masked matMul output', async ({ expect }) => {
        await selectBackend('webgpu');
        const q = randomNormal([100, 3, 1, 64], 0, 1, 'float32');
        const k = randomNormal([100, 3, 64, 64], 0, 1, 'float32');
        const packedQ = pack16(q);
        const packedK = pack16(k);
        const unpackedQ = unpack16(packedQ);
        const unpackedK = unpack16(packedK);

        const rawAttention = matMul(unpackedQ, unpackedK, false, true);
        const maskedAttention = unpack16(attentionMask(packedQ, packedK, 1, 0));

        const repackedRawAttention = unpack16(pack16(rawAttention));

        const rawAttentionData = Array.from(await repackedRawAttention.data());
        const maskedAttentionData = Array.from(await maskedAttention.data());

        // Manually mask the rawAttentionData
        const T1 = q.shape[2]!;
        const T2 = k.shape[2]!;
        for (let b = 0; b < 100; b++) {
            for (let h = 0; h < 3; h++) {
                for (let t1 = 0; t1 < T1; t1++) {
                    for (let t2 = 0; t2 < T2; t2++) {
                        const index = ((b * 3 + h) * T1 + t1) * T2 + t2;
                        if (t2 > t1) {
                            rawAttentionData[index] = Number.NEGATIVE_INFINITY;
                        }
                    }
                }
            }
        }

        const error = arraysClose(rawAttentionData, maskedAttentionData);
        expect(error).toBeLessThan(1e-3);

        expect(maskedAttentionData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
    });
});
