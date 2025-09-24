import { ones, Tensor, tidy, variable, Variable } from '@tensorflow/tfjs-core';
import BaseLayer, { GPTLayerConfig } from './BaseLayer';
import { normRMS } from '@base/ops/normRMS';

export default class RMSNorm extends BaseLayer {
    private gamma: Variable;

    constructor(config: GPTLayerConfig, name = '') {
        super(config);
        this.gamma = variable(ones([config.gpt.nEmbed]), true, `${name}_gamma`, 'float32');
    }

    get trainableWeights(): Variable[] {
        return [this.gamma];
    }

    set trainable(value: boolean) {
        this.gamma.trainable = value;
    }

    getWeights(): Tensor[] {
        return [this.gamma];
    }

    setWeights(weights: Tensor[]): void {
        this.gamma.assign(weights[0]);
    }

    apply(x: Tensor): Tensor {
        return tidy(() => {
            this.startMemory();
            // RMSNorm: x / sqrt(mean(x^2) + epsilon), then scale by gamma
            /*const meanSquare = x.square().mean(-1, true);
            const invRms = meanSquare.add(this.epsilon).rsqrt();
            const normalized = x.mul(invRms);
            const result = normalized.mul(this.gamma);*/
            const result = normRMS(x, this.gamma);
            this.endMemory('RMSNorm');
            return result;
        });
    }

    dispose() {
        this.gamma.dispose();
    }
}
