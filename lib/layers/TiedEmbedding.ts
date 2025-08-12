import type TF from '@tensorflow/tfjs';
import { Initializer } from '@tensorflow/tfjs-layers/dist/initializers';
import { dot } from '@tensorflow/tfjs-layers/dist/backend/tfjs_backend';

export default class TiedEmbeddingOutputLayer {
    private vocabSize: number;
    private embedDim: number;
    private tf: typeof TF;
    private tiedWeights: TF.Variable;
    private initializer: Initializer;

    constructor(tf: typeof TF, config: { vocabSize: number; embedDim: number; name?: string }, name?: string) {
        this.vocabSize = config.vocabSize;
        this.embedDim = config.embedDim;
        this.tf = tf;

        this.initializer = this.tf.initializers.randomNormal({
            mean: 0.0,
            stddev: 0.02,
        });

        this.tiedWeights = this.tf.variable(
            this.initializer.apply([this.vocabSize, this.embedDim]),
            true,
            name || 'tied_embedding'
        );
    }

    get variables(): TF.Variable[] {
        return [this.tiedWeights];
    }

    embed(inputs: TF.Tensor): TF.Tensor {
        return this.tf.gather(this.tiedWeights, inputs, 0);
    }

    project(inputs: TF.Tensor): TF.Tensor {
        return dot(inputs, this.tiedWeights.transpose());
    }

    getWeights(): TF.Tensor[] {
        return [this.tiedWeights];
    }

    setWeights(weights: TF.Tensor[]): void {
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
