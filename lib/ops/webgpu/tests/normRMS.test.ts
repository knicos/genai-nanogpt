import { afterAll, describe, it } from 'vitest';
import { create, globals } from 'webgpu';
import { pack16 } from '../../pack16';
import { unpack16 } from '../../unpack16';
import { arraysClose } from '@base/utilities/arrayClose';

Object.assign(globalThis, globals);
const navigator = { gpu: create([]) };
Object.assign(globalThis.navigator, navigator);

import { selectBackend } from '@base/backend';
import { ones, randomNormal } from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
import { normRMS } from '@base/ops/normRMS';
import { normRMSGradConfig } from '@base/ops/grads/normRMS';
import { isPackedTensor } from '@base/utilities/packed';

describe('RMS Norm 16-bit', { timeout: 30000 }, () => {
    afterAll(() => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
    });
    it('should match the 32-bit version', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([32, 128, 192], 0, 1, 'float32');
        const gamma = ones([192], 'float32');

        const pX = pack16(x);
        const uX = unpack16(pX);

        const packedRMS = normRMS(pX, gamma);
        const originalRMS = unpack16(pack16(normRMS(uX, gamma)));

        const unpackedRMS = unpack16(packedRMS);

        const originalData = await originalRMS.data();
        const unpackedData = await unpackedRMS.data();

        const error = arraysClose(originalData, unpackedData);
        expect(error).toBeLessThan(1e-2);
    });

    it('produces correct norm compared to manual approach', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([32, 128, 192], 0, 1, 'float32');
        const gamma = ones([192], 'float32');

        const fusedRMS = normRMS(x, gamma);

        const meanSquares = x.square().mean(2, true);
        const rms = meanSquares.add(1e-8).sqrt();
        const normalized = x.div(rms);
        const manualRMS = normalized.mul(gamma);

        const manualData = await manualRMS.data();
        const fusedData = await fusedRMS.data();

        const error = arraysClose(manualData, fusedData);
        expect(error).toBeLessThan(1e-4);
    });

    it('has valid gradients', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([64, 32], 0, 1, 'float32');
        const y = randomNormal([64, 32], 0, 1, 'float32');
        const gamma = ones([32], 'float32');

        const packedX = pack16(x);
        const packedY = pack16(y);
        const packedGamma = pack16(gamma);

        const gradX = normRMSGradConfig.gradFunc(packedY, [packedX, packedGamma], { dim: 1 }).x();
        const gradXData = await unpack16(gradX).data();

        expect(gradXData.length).toBe(x.size);
        expect(gradXData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
        expect(gradXData.some((v) => isNaN(v))).toBe(false);
        expect(gradXData.some((v) => !isFinite(v))).toBe(false);
        expect(gradX.dtype).toBe('packedF16');
        expect(isPackedTensor(gradX)).toBe(true);

        const gradGamma = normRMSGradConfig.gradFunc(packedY, [packedX, packedGamma], { dim: 1 }).gamma();
        const gradGammaData = await unpack16(gradGamma).data();

        expect(gradGammaData.length).toBe(gamma.size);
        expect(gradGammaData.every((v) => Math.abs(v) < 1e-8)).toBe(false);
        expect(gradGammaData.some((v) => isNaN(v))).toBe(false);
        expect(gradGammaData.some((v) => !isFinite(v))).toBe(false);
        expect(gradGamma.dtype).toBe('packedF16');
        expect(isPackedTensor(gradGamma)).toBe(true);
    });

    it('produces similar gradients for each precision', async ({ expect }) => {
        await selectBackend('webgpu');
        const x = randomNormal([64, 128], 0, 1, 'float32');
        const y = randomNormal([64, 128], 0, 1, 'float32');
        const gamma = ones([128], 'float32');

        const packedX = pack16(x);
        const packedY = pack16(y);
        const packedGamma = pack16(gamma);

        const gradX16 = unpack16(normRMSGradConfig.gradFunc(packedY, [packedX, packedGamma], { dim: 1 }).x());
        const gradX16Data = await gradX16.data();

        const gradX32 = unpack16(
            pack16(
                normRMSGradConfig
                    .gradFunc(unpack16(packedY), [unpack16(packedX), unpack16(packedGamma)], { dim: 1 })
                    .x()
            )
        );
        const gradX32Data = await gradX32.data();

        const error = arraysClose(gradX16Data, gradX32Data);
        expect(error).toBeLessThan(1e-3);
    });
});
