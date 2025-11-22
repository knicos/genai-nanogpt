import { mod, range, scalar, Tensor, tidy } from '@tensorflow/tfjs-core';
import BaseLayer from './BaseLayer';
import { GPTConfig, ModelForwardAttributes } from '@base/main';
import { layers, initializers } from '@tensorflow/tfjs-layers';
import { add } from '@tensorflow/tfjs-core/dist/engine';

export default class PositionEmbedding extends BaseLayer {
    private wpe?: layers.Layer; // Position embeddings
    private drop: layers.Layer; // Dropout

    constructor(config: GPTConfig, name = '', parent?: BaseLayer) {
        super(config, parent);
        this.wpe = layers.embedding({
            inputDim: this.config.blockSize,
            outputDim: this.config.nEmbed,
            name,
            embeddingsInitializer: initializers.randomNormal({ mean: 0.0, stddev: 0.02 }),
        });
        this.drop = layers.dropout({ rate: this.config.dropout });
    }

    forward(attrs: ModelForwardAttributes, x: Tensor): Tensor {
        const pastLen = attrs.cache?.[0]?.length ?? 0;
        return tidy(() => {
            const [, seqLen] = x.shape;
            const maxCtx = this.config.blockSize;
            // position_ids = (pastLen + arange(T)) % maxCtx    // stays in [0, blockSize)
            const rng = range(0, seqLen, 1, 'int32'); // (t,)
            const posIdx = mod(add(rng, scalar(pastLen, 'int32')), scalar(maxCtx, 'int32')) as Tensor;
            const posEmb = this.wpe!.apply(posIdx) as Tensor; // (b, t, n_embd)

            const embSum = x.add(posEmb);

            const out = this.drop.apply(embSum, { training: attrs.training }) as Tensor;
            return out;
        });
    }
}
