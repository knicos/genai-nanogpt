import type TF from '@tensorflow/tfjs';
import { GPTConfig } from './config';

// Multi-layer perceptron
export default class MLP {
    private cFc: TF.layers.Layer;
    private cProj: TF.layers.Layer;
    private dropout: TF.layers.Layer;
    private tf: typeof TF;
    private index: number;
    private _trainable: boolean = true;

    constructor(tf: typeof TF, index: number, config: GPTConfig) {
        this.tf = tf;
        this.index = index;
        this.cFc = this.tf.layers.dense({
            units: 4 * config.nEmbed,
            activation: 'gelu',
            useBias: config.biasInLinear,
            name: `mlp_hidden_${index}`,
        });

        this.cProj = this.tf.layers.dense({
            units: config.nEmbed,
            useBias: config.biasInLinear,
            name: `mlp_output_${index}`,
        });

        this.dropout = this.tf.layers.dropout({ rate: config.dropout });
    }

    get trainable(): boolean {
        return this._trainable;
    }

    set trainable(value: boolean) {
        this._trainable = value;
        this.cFc.trainable = value;
        this.cProj.trainable = value;
    }

    saveWeights(map: Map<string, TF.Tensor[]>): void {
        map.set(`block_${this.index}_mlpHidden`, this.cFc.getWeights());
        map.set(`block_${this.index}_mlpOut`, this.cProj.getWeights());
    }

    loadWeights(weights: Map<string, TF.Tensor[]>): void {
        this.cFc.setWeights(weights.get(`block_${this.index}_mlpHidden`) || []);
        this.cProj.setWeights(weights.get(`block_${this.index}_mlpOut`) || []);
    }

    call(x: TF.Tensor, training = false): TF.Tensor {
        return this.tf.tidy(() => {
            const hidden = this.cFc.apply(x) as TF.Tensor;
            const projected = this.cProj.apply(hidden) as TF.Tensor;
            return this.dropout.apply(projected, { training }) as TF.Tensor;
        });
    }
}
