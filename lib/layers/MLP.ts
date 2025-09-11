import type TF from '@tensorflow/tfjs';
import { GPTConfig } from '../config';
import BaseLayer from './BaseLayer';

// Multi-layer perceptron
export default class MLP extends BaseLayer {
    private cFc: TF.layers.Layer;
    private cProj: TF.layers.Layer;
    private dropout: TF.layers.Layer;
    private tf: typeof TF;
    private index: number;
    private _trainable: boolean = true;

    constructor(tf: typeof TF, index: number, config: GPTConfig) {
        super();
        this.tf = tf;
        this.index = index;
        this.cFc = this.tf.layers.dense({
            units: config.mlpFactor * config.nEmbed,
            activation: 'gelu',
            useBias: config.biasInLinear,
            kernelInitializer: this.tf.initializers.randomNormal({
                mean: 0.0,
                stddev: 0.02,
            }),
            biasInitializer: 'zeros',
            name: `block_${index}_mlp_cFc`,
        });

        this.cProj = this.tf.layers.dense({
            units: config.nEmbed,
            useBias: config.biasInLinear,
            kernelInitializer: this.tf.initializers.randomNormal({
                mean: 0.0,
                stddev: 0.02 / Math.sqrt(2 * config.nLayer),
            }),
            biasInitializer: 'zeros',
            name: `block_${index}_mlp_cProj`,
        });

        this.dropout = this.tf.layers.dropout({ rate: config.dropout });
    }

    get variables(): TF.Variable[] {
        return [
            ...this.cFc.trainableWeights.map((v) => v.read() as TF.Variable),
            ...this.cProj.trainableWeights.map((v) => v.read() as TF.Variable),
        ];
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
            this.startMemory();
            const hidden = this.cFc.apply(x) as TF.Tensor;
            const projected = this.cProj.apply(hidden) as TF.Tensor;
            const output = this.dropout.apply(projected, { training }) as TF.Tensor;
            this.endMemory('MLP');
            return output;
        });
    }

    dispose() {
        this.cFc.dispose();
        this.cProj.dispose();
        this.dropout.dispose();
    }
}
