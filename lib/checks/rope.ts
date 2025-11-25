import RoPECache from '@base/layers/RoPECache';
import { engine, setBackend, tensor4d } from '@tensorflow/tfjs-core';

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

    const config = {
        biasInLayerNorm: false,
        vocabSize: 20,
        nEmbed: 16,
        nHead: 2,
        nLayer: 1,
        biasInLinear: false,
        dropout: 0.0,
        blockSize: 128,
        mlpFactor: 4,
        useRope: true,
    };
    const ropeCache = new RoPECache(config);
    ropeCache.ensureRopeCache(120);

    // Custom op
    const custom = engine().runKernel(
        'Rope',
        { x, sin: ropeCache.getSin()!, cos: ropeCache.getCos()! },
        { pastLen: 20 }
    );
    if (Array.isArray(custom)) {
        return custom.map((t) => t.array());
    }
    return custom.array();
}
