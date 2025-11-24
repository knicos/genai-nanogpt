import { engine, setBackend, Tensor, tensor2d } from '@tensorflow/tfjs-core';

export async function execute(backend: string) {
    await setBackend(backend);
    const x = tensor2d(
        [
            [0.1, 0.2, 0, 0],
            [0.1, 0.2, 0, 0],
            [0.0, 0.0, 0, 0],
            [0.0, 0.0, 0, 0],
        ],
        [4, 4]
    );

    // Custom op
    const custom = engine().runKernel('Gelu', { x }) as Tensor;
    const customArr = await custom.array();
    return customArr;
}
