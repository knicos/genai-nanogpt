import type TF from '@tensorflow/tfjs';
import BaseLayer from './BaseLayer';

export default class RMSNorm extends BaseLayer {
    private gamma: TF.Variable;
    private epsilon: number;
    private tf: typeof TF;

    constructor(tf: typeof TF, shape: number[], epsilon = 1e-8, name = '') {
        super();
        this.tf = tf;
        this.epsilon = epsilon;
        this.gamma = tf.variable(tf.ones(shape), true, `${name}_gamma`, 'float32');
    }

    get trainableWeights(): TF.Variable[] {
        return [this.gamma];
    }

    set trainable(value: boolean) {
        this.gamma.trainable = value;
    }

    getWeights(): TF.Tensor[] {
        return [this.gamma];
    }

    setWeights(weights: TF.Tensor[]): void {
        this.gamma.assign(weights[0]);
    }

    apply(x: TF.Tensor): TF.Tensor {
        return this.tf.tidy(() => {
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
