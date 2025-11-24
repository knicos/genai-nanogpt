import { engine, setBackend, Tensor, tensor2d, tensor4d } from '@tensorflow/tfjs-core';

export async function execute(backend: string) {
    const pastLen = 0;
    await setBackend(backend);
    const q = tensor4d(
        [
            [
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.3, 0.4, 0.5, 0.6],
                ],
            ],
        ],
        [1, 1, 2, 4]
    ); // (1,1,T,hs)
    const k = tensor4d(
        [
            [
                [
                    [0.5, 0.6, 0.5, 0.6],
                    [0.7, 0.8, 0.7, 0.8],
                ],
            ],
        ],
        [1, 1, 2, 4]
    ); // (1,1,T,hs)
    const mask = tensor2d(
        [
            [0, -Infinity, -Infinity, -Infinity],
            [0, 0, 0, -Infinity],
        ],
        [2, 4]
    ); // (T,T)
    const divisor = 0.5;

    // Custom op
    const custom = engine().runKernel('AttentionMask', { q, k, mask }, { divisor, pastLen }) as Tensor;
    const customArr = await custom.array();
    return customArr;
}
