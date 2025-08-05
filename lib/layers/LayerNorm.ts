import type TF from '@tensorflow/tfjs';

export default class LayerNorm {
    private gamma: TF.Variable;
    //private beta: TF.Variable;
    private epsilon: number;
    private tf: typeof TF;

    constructor(tf: typeof TF, shape: number[], epsilon = 1e-5, name = '') {
        this.tf = tf;
        this.epsilon = epsilon;

        // Initialize gamma (scale) to ones and beta (shift) to zeros
        this.gamma = tf.variable(tf.ones(shape), true, `${name}_gamma`, 'float32');
        //this.beta = tf.variable(tf.zeros(shape), true, `${name}_beta`, 'float32');
    }

    get trainableWeights(): TF.Variable[] {
        return [this.gamma]; //, this.beta];
    }

    set trainable(value: boolean) {
        this.gamma.trainable = value;
        //this.beta.trainable = value;
    }

    getWeights(): TF.Tensor[] {
        return [this.gamma]; //, this.beta];
    }

    setWeights(weights: TF.Tensor[]): void {
        this.gamma.assign(weights[0]);
        //this.beta.assign(weights[1]);
    }

    apply(x: TF.Tensor): TF.Tensor {
        return this.tf.tidy(() => {
            // Calculate mean and variance along the last axis
            const mean = x.mean(-1, true);
            const subMean = x.sub(mean);
            const subMean2 = subMean.square();
            const variance = subMean2.mean(-1, true);

            // Normalize
            const varEpsilon = variance.add(this.epsilon);
            const invStd = varEpsilon.rsqrt();
            const normalized = subMean.mul(invStd);

            // Scale and shift
            const withGamma = normalized.mul(this.gamma);
            return withGamma; //.add(this.beta); Except there is no beta in this version
        });
    }
}
