import { engine, losses, setBackend, Tensor, tensor, tensor1d, valueAndGrads } from '@tensorflow/tfjs-core';

const xA = Array.from({ length: 16 * 128 * 192 }, () => Math.random());
const gammaA = Array.from({ length: 192 }, () => Math.random());
const targetsA = Array.from({ length: 16 * 128 * 192 }, () => Math.random());

export async function execute(backend: string) {
    await setBackend(backend);
    const gammaT = tensor1d(gammaA, 'float32');
    const xT = tensor(xA, [16, 128, 192], 'float32');
    const targets = tensor(targetsA, [16, 128, 192], 'float32');

    // Use valueAndGrads for both value and gradients
    const fn = (x: Tensor, gamma: Tensor) => {
        const result = engine().runKernel('RMSNorm', { x, gamma }) as Tensor;
        return losses.meanSquaredError(result, targets);
    };
    const { value, grads } = valueAndGrads(fn)([xT, gammaT]);

    const valueArr = await value.array();
    const gradXArr = await grads[0].array();
    const gradGammaArr = await grads[1].array();

    return [valueArr, gradXArr, gradGammaArr];
}
