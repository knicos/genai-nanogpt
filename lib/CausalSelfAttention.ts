import type TF from '@tensorflow/tfjs';
import { GPTConfig } from './config';

// Multi-head self-attention implementation
export default class CausalSelfAttention {
    private config: GPTConfig;
    private cAttn: TF.layers.Layer;
    private cProj: TF.layers.Layer;
    private attnDropout: TF.layers.Layer;
    private residDropout: TF.layers.Layer;
    private bias: TF.Tensor;
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
            name: `attention_${index}`,
            //kernelInitializer: 'glorotUniform',
            //biasInitializer: 'zeros',
        });

        // Output projection
        this.cProj = this.tf.layers.dense({
            units: config.nEmbed,
            useBias: config.biasInLinear,
            name: `projection_${index}`,
            //kernelInitializer: 'glorotUniform',
            //biasInitializer: 'zeros',
        });

        // Dropout layers
        this.attnDropout = this.tf.layers.dropout({ rate: config.dropout });
        this.residDropout = this.tf.layers.dropout({ rate: config.dropout });

        // Causal mask to ensure that attention is only applied to the left in the input sequence
        this.bias = this.tf.linalg.bandPart(this.tf.ones([config.blockSize, config.blockSize]), -1, 0);
        this.divisor = this.tf.scalar(Math.sqrt(config.nEmbed / config.nHead)); // Scaling factor for attention scores
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

    call(x: TF.Tensor, training = false): TF.Tensor {
        return this.tf.tidy(() => {
            const [B, T, C] = x.shape; // batch size, sequence length, embedding dimensionality

            // Calculate query, key, values for all heads in batch and move head forward to be the batch dim
            const qkv = this.cAttn.apply(x) as TF.Tensor; // (B, T, 3*C)
            //const [q, k, v] = this.tf.split(qkv, 3, -1); // Each is (B, T, C)

            const C_per_head = C; // Each of q, k, v has dimension C
            const q = qkv.slice([0, 0, 0], [B, T, C_per_head]); // (B, T, C)
            const k = qkv.slice([0, 0, C_per_head], [B, T, C_per_head]); // (B, T, C)
            const v = qkv.slice([0, 0, 2 * C_per_head], [B, T, C_per_head]); // (B, T, C)

            // Reshape for multi-head attention
            const headDim = C / this.config.nHead;
            const qReshaped = this.tf.reshape(q, [B, T, this.config.nHead, headDim]).transpose([0, 2, 1, 3]); // (B, nh, T, hs)
            const kReshaped = this.tf.reshape(k, [B, T, this.config.nHead, headDim]).transpose([0, 2, 1, 3]); // (B, nh, T, hs)
            const vReshaped = this.tf.reshape(v, [B, T, this.config.nHead, headDim]).transpose([0, 2, 1, 3]); // (B, nh, T, hs)

            // Causal self-attention
            const att = this.tf.matMul(qReshaped, kReshaped, false, true).div(this.divisor); // (B, nh, T, T)

            // Apply causal mask
            const mask = this.bias.slice([0, 0], [T, T]);
            const maskedAtt = att.where(mask.equal(1), this.tf.fill(att.shape, -Infinity));
            //const maskedAtt = att.add(mask.sub(this.one).mul(this.smallest)); // Convert 0s to -1e9, 1s to 0

            const attSoftmax = this.tf.softmax(maskedAtt, -1); // (B, nh, T, T)
            const attDropped = this.attnDropout.apply(attSoftmax, { training }) as TF.Tensor;

            // TODO: Export the attention scores

            // Attention applied to values
            const y = this.tf.matMul(attDropped, vReshaped); // (B, nh, T, hs)

            // Re-assemble all head outputs side by side
            const yTransposed = y.transpose([0, 2, 1, 3]); // (B, T, nh, hs)
            const yReshaped = this.tf.reshape(yTransposed, [B, T, C]); // (B, T, C)

            // Output projection
            const output = this.cProj.apply(yReshaped) as TF.Tensor;
            const finalOutput = this.residDropout.apply(output, { training }) as TF.Tensor;

            return finalOutput;
        });
    }
}
