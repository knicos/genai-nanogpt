import { Initializer } from '@tensorflow/tfjs-layers/dist/initializers';
import { initializers } from '@tensorflow/tfjs-layers';
import { Tensor, variable } from '@tensorflow/tfjs-core';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { GPTConfig } from '@base/models/config';
import { dot16 } from '@base/ops/dot16';
import { isPackedTensor } from '@base/utilities/packed';
import { pack16 } from '@base/ops/pack16';

export default class HeadLayer extends BaseLayer {
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

        this.addVariable(this.WEIGHTS, variable(this.initializer.apply([this.embedDim, this.vocabSize]), true, name));
    }

    project(inputs: Tensor): Tensor {
        const packedWeights = isPackedTensor(inputs)
            ? pack16(this.getVariable(this.WEIGHTS), undefined, 32)
            : this.getVariable(this.WEIGHTS);

        const result = dot16(inputs, packedWeights);
        if (isPackedTensor(inputs)) {
            packedWeights.dispose();
        }
        return result;
    }

    forward(_: ForwardAttributes, x: Tensor): Tensor {
        return this.project(x);
    }
}
