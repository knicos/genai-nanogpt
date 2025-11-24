import { engine, setBackend, tensor, Tensor, tensor1d } from '@tensorflow/tfjs-core';

const xA = Array.from({ length: 16 * 128 * 192 }, () => Math.random());
const gammaA = Array.from({ length: 192 }, () => Math.random());
const dyA = Array.from({ length: 16 * 128 * 192 }, () => Math.random());

export async function execute(backend: string) {
    await setBackend(backend);
    const gammaT = tensor1d(gammaA, 'float32');
    const xT = tensor(xA, [16, 128, 192], 'float32');
    const dyT = tensor(dyA, [16, 128, 192], 'float32');
    const result = engine().runKernel('RMSNormGrad', { x: xT, gamma: gammaT, dy: dyT }) as Tensor[];
    const dxArr = await result[0].array();
    const dGamma = await result[1].array();

    return [dxArr, dGamma];
}
