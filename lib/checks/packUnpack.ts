import { engine, setBackend, Tensor, tensor2d } from '@tensorflow/tfjs-core';

export async function execute(backend: string) {
    await setBackend(backend);
    const x = tensor2d(
        [
            [0.1, 0.2, 0, 0, 1230, 1232331234, -12234234],
            [0.1, 0.2, 0, 0, -1230, -1232331234, 12234234],
            [0.0, 0.0, 0, 0, -1, 0, 0],
            [0.0, 0.0, 0, 0, -0.1, 0.001, 0],
        ],
        [4, 7]
    );

    // Custom op
    const packed = engine().runKernel('Pack16', { x }) as Tensor;
    const unpacked = engine().runKernel('Unpack16', { x: packed }) as Tensor;
    const customArr = await unpacked.array();
    return customArr;
}
