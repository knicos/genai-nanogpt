import { engine, setBackend, Tensor, tensor2d, tensor3d } from '@tensorflow/tfjs-core';

export async function execute(backend: string) {
    await setBackend(backend);

    const x = tensor3d(
        [
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
        ],
        [1, 2, 2]
    ); // (1,T,hs)
    const kernel = tensor2d(
        [
            [0.5, 0.6, 0.9, 1.0, 1.3, 1.4],
            [0.7, 0.8, 1.1, 1.2, 1.5, 1.6],
        ],
        [2, 6]
    );

    // Custom op
    const custom = engine().runKernel('QKV', { x, kernel }, { heads: 1 }) as Tensor[];
    const customQ = await custom[0].array();
    const customK = await custom[1].array();
    const customV = await custom[2].array();

    return [customQ, customK, customV];
}
