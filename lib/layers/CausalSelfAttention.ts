import type TF from '@tensorflow/tfjs';
import { GPTConfig } from '../config';

// Multi-head self-attention implementation
export default class CausalSelfAttention {
    private config: GPTConfig;
    private cAttn: TF.layers.Layer;
    private cProj: TF.layers.Layer;
    private attnDropout: TF.layers.Layer;
    private residDropout: TF.layers.Layer;
    private bias: TF.Tensor;
    private maskInf: TF.Tensor;
    private tf: typeof TF;
    private divisor: TF.Tensor;
    private index: number;
    private _trainable: boolean = true;

    constructor(tf: typeof TF, index: number, config: GPTConfig) {
        this.config = config;
        this.tf = tf;
        this.index = index;

        // Key, query, value projections for all heads, but in a batch
        this.cAttn = this.tf.layers.dense({
            units: 3 * config.nEmbed,
            useBias: config.biasInLinear,
            name: `block_${index}_attn_cAttn`,
            kernelInitializer: this.tf.initializers.randomNormal({
                mean: 0.0,
                stddev: 0.02,
            }),
            biasInitializer: 'zeros',
        });

        // Output projection
        this.cProj = this.tf.layers.dense({
            units: config.nEmbed,
            useBias: config.biasInLinear,
            name: `block_${index}_attn_cProj`,
            kernelInitializer: this.tf.initializers.randomNormal({
                mean: 0.0,
                stddev: 0.02 / Math.sqrt(2 * config.nLayer),
            }),
            biasInitializer: 'zeros',
        });

        // Dropout layers
        this.attnDropout = this.tf.layers.dropout({ rate: config.dropout });
        this.residDropout = this.tf.layers.dropout({ rate: config.dropout });

        // Causal mask to ensure that attention is only applied to the left in the input sequence
        this.bias = this.tf.linalg.bandPart(this.tf.ones([config.blockSize, config.blockSize]), -1, 0).cast('bool');
        this.divisor = this.tf.scalar(1 / Math.sqrt(config.nEmbed / config.nHead)); // Scaling factor for attention scores
        this.maskInf = this.tf.zeros([config.blockSize, config.blockSize]).where(this.bias, -Infinity);
    }

    get variables(): TF.Variable[] {
        return [
            ...this.cAttn.trainableWeights.map((v) => v.read() as TF.Variable),
            ...this.cProj.trainableWeights.map((v) => v.read() as TF.Variable),
        ];
    }

    get trainable(): boolean {
        return this._trainable;
    }

    set trainable(value: boolean) {
        this._trainable = value;
        this.cAttn.trainable = value;
        this.cProj.trainable = value;
        //this.attnDropout.trainable = value;
        //this.residDropout.trainable = value;
    }

    saveWeights(map: Map<string, TF.Tensor[]>) {
        map.set(`block_${this.index}_cAttn`, this.cAttn.getWeights());
        map.set(`block_${this.index}_cProj`, this.cProj.getWeights());
    }

    loadWeights(weights: Map<string, TF.Tensor[]>): void {
        this.cAttn.setWeights(weights.get(`block_${this.index}_cAttn`) || []);
        this.cProj.setWeights(weights.get(`block_${this.index}_cProj`) || []);
    }

    private getAttentionScores(q: TF.Tensor, k: TF.Tensor, training: boolean): TF.Tensor {
        const T = q.shape[2]!; // Sequence length

        // Causal self-attention
        const attUnscaled = this.tf.matMul(q, k, false, true); // (B, nh, T, T)
        const att = attUnscaled.mul(this.divisor); // Scale by sqrt(d_k)

        // Apply causal mask
        const mask = this.maskInf.slice([0, 0], [T, T]);
        const maskedAtt = att.add(mask);

        const attSoftmax = this.tf.softmax(maskedAtt, -1); // (B, nh, T, T)
        return this.attnDropout.apply(attSoftmax, { training }) as TF.Tensor;
    }

    private getQKV(x: TF.Tensor): [TF.Tensor, TF.Tensor, TF.Tensor] {
        const [B, T, C] = x.shape; // batch size, sequence length, embedding dimensionality

        // Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        const qkv = this.cAttn.apply(x) as TF.Tensor; // (B, T, 3*C)
        //x.dispose();
        const [q, k, v] = this.tf.split(qkv, 3, -1); // Each is (B, T, C)
        qkv.dispose();

        // Reshape for multi-head attention
        const headDim = C / this.config.nHead;

        const qReshaped = this.tf.reshape(q, [B, T, this.config.nHead, headDim]);
        q.dispose();
        const qT = qReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
        qReshaped.dispose();

        const kReshaped = this.tf.reshape(k, [B, T, this.config.nHead, headDim]);
        k.dispose();
        const kT = kReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
        kReshaped.dispose();

        const vReshaped = this.tf.reshape(v, [B, T, this.config.nHead, headDim]);
        v.dispose();
        const vT = vReshaped.transpose([0, 2, 1, 3]); // (B, nh, T, hs)
        vReshaped.dispose();
        return [qT, kT, vT];
    }

    private getOutputProjection(x: TF.Tensor, training: boolean): TF.Tensor {
        const B = x.shape[0]!; // batch size
        const T = x.shape[2]!; // sequence length
        const C = this.config.nEmbed; // embedding dimensionality

        // Re-assemble all head outputs side by side
        const yTransposed = x.transpose([0, 2, 1, 3]); // (B, T, nh, hs)
        const yReshaped = this.tf.reshape(yTransposed, [B, T, C]); // (B, T, C)

        // Output projection
        const output = this.cProj.apply(yReshaped) as TF.Tensor;
        const finalOutput = this.residDropout.apply(output, { training }) as TF.Tensor;
        return finalOutput;
    }

    call(x: TF.Tensor, training = false, includeAttention = false): { output: TF.Tensor; attention?: TF.Tensor } {
        return this.tf.tidy(() => {
            const [q, k, v] = this.getQKV(x);
            const attScores = this.getAttentionScores(q, k, training); // (B, nh, T, T)

            // Attention applied to values
            const y = this.tf.matMul(attScores, v); // (B, nh, T, hs)

            const output = this.getOutputProjection(y, training); // (B, T, C)
            return { output, attention: includeAttention ? attScores.mean(1) : undefined };
        });
    }

    dispose() {
        this.cAttn.dispose();
        this.cProj.dispose();
        this.attnDropout.dispose();
        this.residDropout.dispose();
        this.bias.dispose();
        this.maskInf.dispose();
        this.divisor.dispose();
    }
}
