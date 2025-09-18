import { Tensor, tidy, Variable } from '@tensorflow/tfjs-core';
import { GPTConfig } from '../config';
import BaseLayer from './BaseLayer';
import { layers, initializers } from '@tensorflow/tfjs-layers';

// Multi-layer perceptron
export default class MLP extends BaseLayer {
    private cFc: layers.Layer;
    private cProj: layers.Layer;
    private dropout: layers.Layer;
    private index: number;
    private _trainable: boolean = true;

    constructor(index: number, config: GPTConfig) {
        super();
        this.index = index;
        this.cFc = layers.dense({
            units: config.mlpFactor * config.nEmbed,
            activation: 'gelu',
            useBias: config.biasInLinear,
            kernelInitializer: initializers.randomNormal({
                mean: 0.0,
                stddev: 0.02,
            }),
            biasInitializer: 'zeros',
            name: `block_${index}_mlp_cFc`,
        });

        this.cProj = layers.dense({
            units: config.nEmbed,
            useBias: config.biasInLinear,
            kernelInitializer: initializers.randomNormal({
                mean: 0.0,
                stddev: 0.02 / Math.sqrt(2 * config.nLayer),
            }),
            biasInitializer: 'zeros',
            name: `block_${index}_mlp_cProj`,
        });

        this.dropout = layers.dropout({ rate: config.dropout });
    }

    get variables(): Variable[] {
        return [
            ...this.cFc.trainableWeights.map((v) => v.read() as Variable),
            ...this.cProj.trainableWeights.map((v) => v.read() as Variable),
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

    saveWeights(map: Map<string, Tensor[]>): void {
        map.set(`block_${this.index}_mlpHidden`, this.cFc.getWeights());
        map.set(`block_${this.index}_mlpOut`, this.cProj.getWeights());
    }

    loadWeights(weights: Map<string, Tensor[]>): void {
        this.cFc.setWeights(weights.get(`block_${this.index}_mlpHidden`) || []);
        this.cProj.setWeights(weights.get(`block_${this.index}_mlpOut`) || []);
    }

    call(x: Tensor, training = false): Tensor {
        return tidy(() => {
            this.startMemory();
            const hidden = this.cFc.apply(x) as Tensor;
            const projected = this.cProj.apply(hidden) as Tensor;
            const output = this.dropout.apply(projected, { training }) as Tensor;
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
