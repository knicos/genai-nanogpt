import { ones, Tensor, tidy, variable, Variable } from '@tensorflow/tfjs-core';
import BaseLayer from './BaseLayer';

export default class RMSNorm extends BaseLayer {
    private gamma: Variable;
    private epsilon: number;

    constructor(shape: number[], epsilon = 1e-8, name = '') {
        super();
        this.epsilon = epsilon;
        this.gamma = variable(ones(shape), true, `${name}_gamma`, 'float32');
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
            const meanSquare = x.square().mean(-1, true);
            const invRms = meanSquare.add(this.epsilon).rsqrt();
            const normalized = x.mul(invRms);
            const result = normalized.mul(this.gamma);
            this.endMemory('RMSNorm');
            return result;
        });
    }

    dispose() {
        this.gamma.dispose();
    }
}
