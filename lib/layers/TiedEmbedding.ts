import { Initializer } from '@tensorflow/tfjs-layers/dist/initializers';
import { initializers } from '@tensorflow/tfjs-layers';
import { gather, Tensor, variable, Variable } from '@tensorflow/tfjs-core';
import { dot } from '@tensorflow/tfjs-layers/dist/backend/tfjs_backend';

export default class TiedEmbeddingOutputLayer {
    private vocabSize: number;
    private embedDim: number;
    private tiedWeights: Variable;
    private initializer: Initializer;

    constructor(config: { vocabSize: number; embedDim: number; name?: string }, name?: string) {
        this.vocabSize = config.vocabSize;
        this.embedDim = config.embedDim;

        this.initializer = initializers.randomNormal({
            mean: 0.0,
            stddev: 0.02,
        });

        this.tiedWeights = variable(
            this.initializer.apply([this.vocabSize, this.embedDim]),
            true,
            name || 'tied_embedding'
        );
    }

    get variables(): Variable[] {
        return [this.tiedWeights];
    }

    embed(inputs: Tensor): Tensor {
        return gather(this.tiedWeights, inputs, 0);
    }

    project(inputs: Tensor): Tensor {
        return dot(inputs, this.tiedWeights.transpose());
    }

    getWeights(): Tensor[] {
        return [this.tiedWeights];
    }

    setWeights(weights: Tensor[]): void {
        this.tiedWeights.assign(weights[0]);
    }

    getConfig() {
        return {
            vocabSize: this.vocabSize,
            embedDim: this.embedDim,
        };
    }

    dispose() {
        this.tiedWeights.dispose();
    }
}
