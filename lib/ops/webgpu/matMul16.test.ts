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
import { matMul, mul, randomNormal, scalar } from '@tensorflow/tfjs-core';
import { matMul16, matMul16Scaled } from '../matMul16';
import { matMul16GradConfig } from '../grads/matMul16';
import { isPackedTensor } from '@base/utilities/packed';

describe('MatMul 16-bit', { timeout: 30000 }, () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });
    it('produce correct output without transpose', async ({ expect }) => {
        await selectBackend('webgpu');
        const scores = randomNormal([100, 3, 128, 64], 0, 1, 'float32');
        const values = randomNormal([100, 3, 64, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues);
        const packedMatMul = matMul16(packedScores, packedValues);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        expect(repackedRaw.shape).toEqual(unpackedMatMul.shape);

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('produce correct output with transposeA', async ({ expect }) => {
        await selectBackend('webgpu');
        const scores = randomNormal([100, 3, 64, 128], 0, 1, 'float32');
        const values = randomNormal([100, 3, 64, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues, true, false);
        const packedMatMul = matMul16(packedScores, packedValues, true, false);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('produce correct output with transposeB', async ({ expect }) => {
        await selectBackend('webgpu');
        const scores = randomNormal([100, 3, 128, 32], 0, 1, 'float32');
        const values = randomNormal([100, 3, 64, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues, false, true);
        const packedMatMul = matMul16(packedScores, packedValues, false, true);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('can support rank 3', async ({ expect }) => {
        await selectBackend('webgpu');
        const scores = randomNormal([100, 64, 128], 0, 1, 'float32');
        const values = randomNormal([100, 128, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues);
        const packedMatMul = matMul16(packedScores, packedValues);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('can support rank 2', async ({ expect }) => {
        await selectBackend('webgpu');
        const scores = randomNormal([64, 64], 0, 1, 'float32');
        const values = randomNormal([64, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues);
        const packedMatMul = matMul16(packedScores, packedValues);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('can support different ranks', async ({ expect }) => {
        await selectBackend('webgpu');
        const scores = randomNormal([100, 128, 64], 0, 1, 'float32');
        const values = randomNormal([64, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues);
        const packedMatMul = matMul16(packedScores, packedValues);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('can support different ranks with transposeA', async ({ expect }) => {
        await selectBackend('webgpu');
        const scores = randomNormal([100, 128, 64], 0, 1, 'float32');
        const values = randomNormal([128, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues, true, false);
        const packedMatMul = matMul16(packedScores, packedValues, true, false);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        console.log('Out shapes', repackedRaw.shape, unpackedMatMul.shape);

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('can support different ranks with transposeB', async ({ expect }) => {
        await selectBackend('webgpu');
        const scores = randomNormal([100, 64, 64], 0, 1, 'float32');
        const values = randomNormal([128, 64], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues, false, true);
        const packedMatMul = matMul16(packedScores, packedValues, false, true);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('can support broadcasting batch dimensions', async ({ expect }) => {
        await selectBackend('webgpu');
        // A: [1, 64, 64], B: [100, 64, 32] -> broadcast A over batch
        const scores = randomNormal([1, 64, 64], 0, 1, 'float32');
        const values = randomNormal([100, 64, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues);
        const packedMatMul = matMul16(packedScores, packedValues);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('can support broadcasting batch dimensions with transposeA', async ({ expect }) => {
        await selectBackend('webgpu');
        // A: [1, 64, 64], B: [100, 64, 32] -> broadcast A over batch, transposeA
        const scores = randomNormal([1, 64, 64], 0, 1, 'float32');
        const values = randomNormal([100, 64, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues, true, false);
        const packedMatMul = matMul16(packedScores, packedValues, true, false);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('can support broadcasting batch dimensions with transposeB', async ({ expect }) => {
        await selectBackend('webgpu');
        // A: [100, 64, 32], B: [1, 32, 32] -> broadcast B over batch, transposeB
        const scores = randomNormal([100, 64, 32], 0, 1, 'float32');
        const values = randomNormal([1, 32, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues, false, true);
        const packedMatMul = matMul16(packedScores, packedValues, false, true);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });

    it('can support rank 4 tensors', async ({ expect }) => {
        await selectBackend('webgpu');
        // [batch, heads, seq, head_size]
        const scores = randomNormal([2, 3, 64, 64], 0, 1, 'float32');
        const values = randomNormal([2, 3, 64, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = matMul(unpackedScores, unpackedValues);
        const packedMatMul = matMul16(packedScores, packedValues);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);

        expect(unpackedMatMulData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
    });

    it('has valid gradients', async ({ expect }) => {
        await selectBackend('webgpu');
        const A = randomNormal([64, 32], 0, 1, 'float32');
        const B = randomNormal([32, 128], 0, 1, 'float32');

        const packedA = pack16(A);
        const packedB = pack16(B);
        const packedMatMul = matMul16(packedA, packedB);

        const grads = matMul16GradConfig.gradFunc(packedMatMul, [packedA, packedB], {});
        const gradA = grads.A();
        const gradB = grads.B();
        const gradAUnpacked = unpack16(gradA);
        const gradBUnpacked = unpack16(gradB);

        const gradAData = await gradAUnpacked.data();
        const gradBData = await gradBUnpacked.data();

        expect(gradA.size).toBe(packedA.size);
        expect(gradAData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
        expect(gradAData.some((v) => isNaN(v))).toBe(false);
        expect(gradAData.some((v) => !isFinite(v))).toBe(false);
        expect(gradA.dtype).toBe('int32');
        expect(isPackedTensor(gradA)).toBe(true);

        expect(gradB.size).toBe(packedB.size);
        expect(gradBData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
        expect(gradBData.some((v) => isNaN(v))).toBe(false);
        expect(gradBData.some((v) => !isFinite(v))).toBe(false);
        expect(gradB.dtype).toBe('int32');
        expect(isPackedTensor(gradB)).toBe(true);
    });

    it('can scale outputs', async ({ expect }) => {
        await selectBackend('webgpu');
        const scores = randomNormal([100, 3, 128, 64], 0, 1, 'float32');
        const values = randomNormal([100, 3, 64, 32], 0, 1, 'float32');

        const packedScores = pack16(scores);
        const packedValues = pack16(values);

        const unpackedScores = unpack16(packedScores);
        const unpackedValues = unpack16(packedValues);

        const rawMatMul = mul(matMul(unpackedScores, unpackedValues), scalar(2.0, 'float32'));
        const packedMatMul = matMul16Scaled(packedScores, packedValues, 2.0);
        const unpackedMatMul = unpack16(packedMatMul);

        const repackedRaw = unpack16(pack16(rawMatMul));

        const rawMatMulData = await repackedRaw.data();
        const unpackedMatMulData = await unpackedMatMul.data();

        expect(repackedRaw.shape).toEqual(unpackedMatMul.shape);

        const error = arraysClose(rawMatMulData, unpackedMatMulData);
        expect(error).toBeLessThan(1e-3);
    });
});
