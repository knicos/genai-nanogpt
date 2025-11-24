import { engine, setBackend, Tensor, tensor4d } from '@tensorflow/tfjs-core';

export async function execute(backend: string) {
    await setBackend(backend);
    const cache = tensor4d(
        [
            [
                [
                    [0.1, 0.2, 0, 0],
                    [0.1, 0.2, 0, 0],
                    [0.0, 0.0, 0, 0],
                    [0.0, 0.0, 0, 0],
                ],
            ],
        ],
        [1, 1, 4, 4]
    );
    const x = tensor4d([[[[0.1, 0.2, 0.3, 0.4]]]], [1, 1, 1, 4]);

    // Custom op
    const custom = engine().runKernel('AppendCache', { cache, item: x }, { maxSize: 4, pastLen: 2 }) as Tensor;
    const customArr = await custom.array();
    return customArr;
}
