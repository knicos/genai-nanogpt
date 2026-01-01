import { Initializer } from '@tensorflow/tfjs-layers/dist/initializers';
import { initializers } from '@tensorflow/tfjs-layers';
import { gather, Tensor, variable } from '@tensorflow/tfjs-core';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { GPTConfig } from '@base/models/config';
import { dot16 } from '@base/ops/dot16';
import { isPackedTensor } from '@base/utilities/packed';
import { pack16 } from '@base/ops/pack16';
import { transpose16 } from '@base/ops/transpose16';

export default class TiedEmbeddingOutputLayer extends BaseLayer {
    private vocabSize: number;
    private embedDim: number;
    private initializer: Initializer;
    private WEIGHTS: string;

    constructor(config: GPTConfig, name: string, parent?: BaseLayer) {
        super(config, parent);
        this.WEIGHTS = name;
        this.vocabSize = config.vocabSize;
        this.embedDim = config.nEmbed;

        this.initializer = initializers.randomNormal({
            mean: 0.0,
            stddev: 0.02,
        });

        this.addVariable(this.WEIGHTS, variable(this.initializer.apply([this.vocabSize, this.embedDim]), true, name));
    }

    embed(inputs: Tensor): Tensor {
        return gather(this.getVariable(this.WEIGHTS), inputs, 0);
    }

    project(inputs: Tensor): Tensor {
        const packedWeights = isPackedTensor(inputs)
            ? pack16(this.getVariable(this.WEIGHTS), undefined, 32)
            : this.getVariable(this.WEIGHTS);
        const transposedWeights = transpose16(packedWeights);
        if (isPackedTensor(inputs)) {
            packedWeights.dispose();
        }
        const result = dot16(inputs, transposedWeights);
        return result;
    }

    // Dummy, should not be used.
    forward(_: ForwardAttributes, x: Tensor): Tensor {
        return this.project(x);
    }
}
