import { Initializer } from '@tensorflow/tfjs-layers/dist/initializers';
import { initializers } from '@tensorflow/tfjs-layers';
import { gather, Tensor, variable } from '@tensorflow/tfjs-core';
import { dot } from '@tensorflow/tfjs-layers/dist/backend/tfjs_backend';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { GPTConfig } from '@base/models/config';

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
        return dot(inputs, this.getVariable(this.WEIGHTS).transpose());
    }

    // Dummy, should not be used.
    forward(_: ForwardAttributes, x: Tensor): Tensor {
        return this.project(x);
    }
}
