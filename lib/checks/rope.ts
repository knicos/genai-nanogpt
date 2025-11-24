import { engine, setBackend, tensor3d, tensor4d } from '@tensorflow/tfjs-core';

export async function execute(backend: string) {
    await setBackend(backend);

    const x = tensor4d(
        [
            [
                [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ],
            ],
        ],
        [1, 1, 2, 2]
    ); // (1,1,T,rdim)
    const sin = tensor3d([0.5, 0.6], [2, 1, 1]);
    const cos = tensor3d([0.9, 1.0], [2, 1, 1]);

    // Custom op
    const custom = engine().runKernel('Rope', { x, sin, cos }, { pastLen: 0 });
    if (Array.isArray(custom)) {
        return custom.map((t) => t.array());
    }
    return custom.array();
}
