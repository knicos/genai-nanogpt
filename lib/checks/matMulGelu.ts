import { engine, setBackend, Tensor, tensor2d } from '@tensorflow/tfjs';

export async function execute(backend: string) {
    await setBackend(backend);
    const kernel = tensor2d(
        [
            [0.1, 0.2, 9, 10, 11],
            [0.3, 0.4, -9, -10, -11],
            [0.3, 0.4, -9, -10, -11],
            [0.3, 0.4, -9, -10, -11],
            [0.3, 0.4, -9, -10, -11],
        ],
        [5, 5]
    ); // (1,T,hs)
    const x = tensor2d(
        [
            [0.5, 0.6, 70000, -8000, 0],
            [0.7, 0.8, -70000, 80000, 0],
            [0.7, 0.8, -70000, 80000, 0],
            [0.7, 0.8, -70000, 80000, 0],
            [0.7, 0.8, -70000, 80000, 0],
        ],
        [5, 5]
    );

    const custom = engine().runKernel('MatMulGelu', { x, kernel }) as Tensor;
    const customArray = await custom.array();
    return customArray;
}
