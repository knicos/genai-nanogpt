import { afterAll, describe, it } from 'vitest';
import { create, globals } from 'webgpu';
import { pack16 } from '../../pack16';
import { unpack16 } from '../../unpack16';
import { arraysClose } from '@base/utilities/arrayClose';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';
import { getGradient, matMul, mul, randomNormal, reshape, scalar, transpose } from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/register_all_gradients';
import { matMul16, matMul16Gelu, matMul16Scaled } from '../../matMul16';
import { matMul16GradConfig } from '../../grads/matMul16';
import { isPackedTensor } from '@base/utilities/packed';
import { matMulGelu } from '../../matMulGelu';

describe('MatMul 16-bit', { timeout: 30000 }, () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });

    describe('Untransposed ranks', () => {
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

        it('can split last dimension', async ({ expect }) => {
            await selectBackend('webgpu');
            // [batch, heads, seq, head_size]
            const scores = randomNormal([12, 64, 64], 0, 1, 'float32');
            const values = randomNormal([12, 64, 32], 0, 1, 'float32');

            const packedScores = pack16(scores);
            const packedValues = pack16(values);

            const unpackedScores = unpack16(packedScores);
            const unpackedValues = unpack16(packedValues);

            const outputShape = [12, 64, 2, 16 / 2];

            const rawMatMul = matMul(unpackedScores, unpackedValues);
            const packedMatMul = matMul16(packedScores, packedValues, false, false, { forceOutputShape: outputShape });
            const unpackedMatMul = unpack16(packedMatMul);

            const repackedRaw = unpack16(pack16(rawMatMul));

            const rawMatMulData = await repackedRaw.data();
            const unpackedMatMulData = await unpackedMatMul.data();

            const error = arraysClose(rawMatMulData, unpackedMatMulData);
            expect(error).toBeLessThan(1e-3);

            expect(unpackedMatMulData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
        });

        it('can split last dimension and permute', async ({ expect }) => {
            await selectBackend('webgpu');
            // [batch, heads, seq, head_size]
            const scores = randomNormal([12, 64, 64], 0, 1, 'float32');
            const values = randomNormal([12, 64, 32], 0, 1, 'float32');

            const packedScores = pack16(scores);
            const packedValues = pack16(values);

            const unpackedScores = unpack16(packedScores);
            const unpackedValues = unpack16(packedValues);

            const outputShape = [12, 64, 2, 16 / 2];
            const perm = [0, 2, 1, 3];

            const rawMatMul = transpose(reshape(matMul(unpackedScores, unpackedValues), [12, 64, 2, 16]), perm);
            const packedMatMul = matMul16(packedScores, packedValues, false, false, {
                forceOutputShape: outputShape,
                perm,
            });
            const unpackedMatMul = unpack16(packedMatMul);

            console.log('Output shapes:', rawMatMul.shape, unpackedMatMul.shape);

            const repackedRaw = unpack16(pack16(rawMatMul));

            const rawMatMulData = await repackedRaw.data();
            const unpackedMatMulData = await unpackedMatMul.data();

            const error = arraysClose(rawMatMulData, unpackedMatMulData);
            expect(error).toBeLessThan(1e-3);

            expect(unpackedMatMulData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
        });
    });

    describe('Transposed cases', () => {
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
    });

    describe('Broadcasting and higher ranks', () => {
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
    });

    describe('Gradients', () => {
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
            expect(gradA.dtype).toBe('packedF16');
            expect(isPackedTensor(gradA)).toBe(true);

            expect(gradB.size).toBe(packedB.size);
            expect(gradBData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
            expect(gradBData.some((v) => isNaN(v))).toBe(false);
            expect(gradBData.some((v) => !isFinite(v))).toBe(false);
            expect(gradB.dtype).toBe('packedF16');
            expect(isPackedTensor(gradB)).toBe(true);
        });

        it('has valid transposeA gradients', async ({ expect }) => {
            await selectBackend('webgpu');
            const A = randomNormal([32, 64], 0, 1, 'float32');
            const B = randomNormal([32, 128], 0, 1, 'float32');

            const packedA = pack16(A);
            const packedB = pack16(B);
            const packedMatMul = matMul16(packedA, packedB, true, false);

            const grads = matMul16GradConfig.gradFunc(packedMatMul, [packedA, packedB], {
                transposeA: true,
                transposeB: false,
            });
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
            expect(gradA.dtype).toBe('packedF16');
            expect(isPackedTensor(gradA)).toBe(true);

            expect(gradB.size).toBe(packedB.size);
            expect(gradBData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
            expect(gradBData.some((v) => isNaN(v))).toBe(false);
            expect(gradBData.some((v) => !isFinite(v))).toBe(false);
            expect(gradB.dtype).toBe('packedF16');
            expect(isPackedTensor(gradB)).toBe(true);
        });

        it('has valid transposeB gradients', async ({ expect }) => {
            await selectBackend('webgpu');
            const A = randomNormal([64, 32], 0, 1, 'float32');
            const B = randomNormal([128, 32], 0, 1, 'float32');

            const packedA = pack16(A);
            const packedB = pack16(B);
            const packedMatMul = matMul16(packedA, packedB, false, true);

            const grads = matMul16GradConfig.gradFunc(packedMatMul, [packedA, packedB], {
                transposeA: false,
                transposeB: true,
            });
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
            expect(gradA.dtype).toBe('packedF16');
            expect(isPackedTensor(gradA)).toBe(true);

            expect(gradB.size).toBe(packedB.size);
            expect(gradBData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
            expect(gradBData.some((v) => isNaN(v))).toBe(false);
            expect(gradBData.some((v) => !isFinite(v))).toBe(false);
            expect(gradB.dtype).toBe('packedF16');
            expect(isPackedTensor(gradB)).toBe(true);
        });

        it('has correct grad', async ({ expect }) => {
            await selectBackend('webgpu');
            const A = randomNormal([64, 32], 0, 1, 'float32');
            const B = randomNormal([32, 128], 0, 1, 'float32');
            const gradY = randomNormal([64, 128], 0, 1, 'float32');
            const packedGradY = pack16(gradY);

            const packedA = pack16(A);
            const packedB = pack16(B);
            const uA = unpack16(packedA);
            const uB = unpack16(packedB);
            // const unpackedMatMul = matMul(uA, uB);

            const matMulGradFunc = getGradient('BatchMatMul').gradFunc;

            const grads = matMul16GradConfig.gradFunc(packedGradY, [packedA, packedB], {});
            const gradA = grads.A();
            const gradAUnpacked = unpack16(gradA);

            const gradsReal = unpack16(pack16(matMulGradFunc(unpack16(packedGradY), [uA, uB], {}).a()));

            const gradAData = await gradAUnpacked.data();
            const gradARealData = await gradsReal.data();

            const error = arraysClose(gradAData, gradARealData);
            expect(error).toBeLessThan(1e-3);
        });
    });

    describe('Special features', () => {
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

        it('has correct grad for scaled outputs', async ({ expect }) => {
            await selectBackend('webgpu');
            const A = randomNormal([64, 32], 0, 1, 'float32');
            const B = randomNormal([32, 128], 0, 1, 'float32');
            const gradY = randomNormal([64, 128], 0, 1, 'float32');
            const packedGradY = pack16(gradY);

            const packedA = pack16(A);
            const packedB = pack16(B);
            const unpackedMatMul = matMul(A, B);

            const mulGradFunc = getGradient('Multiply').gradFunc;
            const matMulGradFunc = getGradient('BatchMatMul').gradFunc;

            const grads = matMul16GradConfig.gradFunc(packedGradY, [packedA, packedB], { scale: 2.0 });
            const gradA = grads.A();
            const gradAUnpacked = unpack16(gradA);

            const gradsRealMul = mulGradFunc(unpack16(packedGradY), [unpackedMatMul, scalar(2.0, 'float32')], {}).a();
            const gradsReal = unpack16(
                pack16(matMulGradFunc(gradsRealMul, [unpack16(packedA), unpack16(packedB)], {}).a())
            );

            const gradAData = await gradAUnpacked.data();
            const gradARealData = await gradsReal.data();

            const error = arraysClose(gradAData, gradARealData);
            expect(error).toBeLessThan(1e-3);
        });

        it('supports gelu activation', async ({ expect }) => {
            await selectBackend('webgpu');
            const scores = randomNormal([100, 3, 128, 64], 0, 1, 'float32');
            const values = randomNormal([100, 3, 64, 32], 0, 1, 'float32');

            const packedScores = pack16(scores);
            const packedValues = pack16(values);

            const unpackedScores = unpack16(packedScores);
            const unpackedValues = unpack16(packedValues);

            const rawMatMul = matMulGelu(unpackedScores, unpackedValues);
            const packedMatMul = matMul16Gelu(packedScores, packedValues);
            const unpackedMatMul = unpack16(packedMatMul);

            const repackedRaw = unpack16(pack16(rawMatMul));

            const rawMatMulData = await repackedRaw.data();
            const unpackedMatMulData = await unpackedMatMul.data();

            expect(repackedRaw.shape).toEqual(unpackedMatMul.shape);

            const error = arraysClose(rawMatMulData, unpackedMatMulData);
            expect(error).toBeLessThan(1e-3);
        });

        it('supports causal mask', async ({ expect }) => {
            await selectBackend('webgpu');
            const A = randomNormal([100, 3, 128, 256], 0, 1, 'float32');
            const B = randomNormal([100, 3, 256, 128], 0, 1, 'float32');

            const packedA = pack16(A);
            const packedB = pack16(B);

            const packedMatMul = matMul16(packedA, packedB, false, false, { causalMask: true, pastLen: 0 });
            const unpackedMatMul = unpack16(packedMatMul);
            const realMatMul = matMul(unpack16(packedA), unpack16(packedB));

            const repackedReal = unpack16(pack16(realMatMul));

            const repackedRealData = await repackedReal.data();
            const unpackedMatMulData = await unpackedMatMul.data();

            expect(unpackedMatMul.shape).toEqual([100, 3, 128, 128]);

            let wasAllMasked = true;
            let wasCorrect = true;

            // Check that the upper triangular part is -infinity (masked out)
            for (let batch = 0; batch < 100; batch++) {
                for (let head = 0; head < 3; head++) {
                    for (let i = 0; i < 128; i++) {
                        for (let j = 0; j < 128; j++) {
                            const index = batch * 3 * 128 * 128 + head * 128 * 128 + i * 128 + j;
                            if (j > i) {
                                // Should be masked
                                if (unpackedMatMulData[index] !== -Infinity) {
                                    wasAllMasked = false;
                                }
                            } else {
                                // Should match the real matmul
                                if (Math.abs(unpackedMatMulData[index] - repackedRealData[index]) > 1e-3) {
                                    wasCorrect = false;
                                }
                            }
                        }
                    }
                }
            }

            expect(wasAllMasked).toBe(true);
            expect(wasCorrect).toBe(true);
        });

        it('supports causal mask with non-zero past length', async ({ expect }) => {
            const PASTLEN = 90;
            await selectBackend('webgpu');
            const A = randomNormal([100, 3, 128, 256], 0, 1, 'float32');
            const B = randomNormal([100, 3, 256, 128], 0, 1, 'float32');

            const packedA = pack16(A);
            const packedB = pack16(B);

            const packedMatMul = matMul16(packedA, packedB, false, false, { causalMask: true, pastLen: PASTLEN });
            const unpackedMatMul = unpack16(packedMatMul);
            const realMatMul = matMul(unpack16(packedA), unpack16(packedB));

            const repackedReal = unpack16(pack16(realMatMul));

            const repackedRealData = await repackedReal.data();
            const unpackedMatMulData = await unpackedMatMul.data();

            expect(unpackedMatMul.shape).toEqual([100, 3, 128, 128]);

            let wasAllMasked = true;
            let wasCorrect = true;

            // Check that the upper triangular part is -infinity (masked out)
            for (let batch = 0; batch < 100; batch++) {
                for (let head = 0; head < 3; head++) {
                    for (let i = 0; i < 128; i++) {
                        for (let j = 0; j < 128; j++) {
                            const index = batch * 3 * 128 * 128 + head * 128 * 128 + i * 128 + j;
                            if (j > i + PASTLEN) {
                                // Should be masked
                                if (unpackedMatMulData[index] !== -Infinity) {
                                    wasAllMasked = false;
                                }
                            } else {
                                // Should match the real matmul
                                if (Math.abs(unpackedMatMulData[index] - repackedRealData[index]) > 1e-3) {
                                    wasCorrect = false;
                                }
                            }
                        }
                    }
                }
            }

            expect(wasAllMasked).toBe(true);
            expect(wasCorrect).toBe(true);
        });
    });
});
