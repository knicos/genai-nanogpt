import type TF from '@tensorflow/tfjs';
import { GPTConfig } from '../config';

export default class RoPECache {
    public readonly rotaryDim: number;
    private ropeBase: number;
    private ropeInvFreq: TF.Tensor;
    private ropeCos: TF.Tensor | null = null; // [cacheLen, rotaryDim/2]
    private ropeSin: TF.Tensor | null = null; // [cacheLen, rotaryDim/2]
    private ropeCacheLen = 0;

    constructor(private readonly tf: typeof TF, private readonly config: GPTConfig) {
        const headDim = this.config.nEmbed / this.config.nHead;
        this.rotaryDim = headDim;
        if (this.rotaryDim % 2 !== 0) {
            throw new Error('rotaryDim must be even');
        }
        this.ropeBase = 10000; // Could be a little smaller for shorter sequences
        const i = this.tf.range(0, this.rotaryDim, 2, 'float32'); // even indices
        const exponent = i.div(this.tf.scalar(this.rotaryDim, 'float32')); // i/rotaryDim
        const basePow = this.tf.pow(this.tf.scalar(this.ropeBase, 'float32'), exponent);
        this.ropeInvFreq = this.tf.reciprocal(basePow); // [rotaryDim/2]

        exponent.dispose();
        basePow.dispose();
        i.dispose();

        if (this.config.useRope === false) {
            this.ropeCos = null;
            this.ropeSin = null;
            this.ropeCacheLen = 0;
        } else {
            this.tf.tidy(() => {
                this.ensureRopeCache(this.config.blockSize * 4);
            });
        }
    }

    public ensureRopeCache(needed: number) {
        if (needed <= this.ropeCacheLen) return;
        if (this.ropeCos) this.ropeCos.dispose();
        if (this.ropeSin) this.ropeSin.dispose();
        const nextSize = Math.max(needed, this.ropeCacheLen + this.config.blockSize * 4);
        const positions = this.tf.range(0, nextSize, 1, 'float32').expandDims(1); // [L,1]
        const freqs = positions.mul(this.ropeInvFreq.expandDims(0)); // [L, rd/2]
        this.ropeCos = this.tf.keep(this.tf.cos(freqs).expandDims(-1)); // [L, rd/2, 1]
        this.ropeSin = this.tf.keep(this.tf.sin(freqs).expandDims(-1)); // [L, rd/2, 1]
        this.ropeCacheLen = nextSize;
    }

    public getCos(): TF.Tensor | null {
        return this.ropeCos;
    }

    public getSin(): TF.Tensor | null {
        return this.ropeSin;
    }

    public dispose() {
        if (this.ropeCos) this.ropeCos.dispose();
        if (this.ropeSin) this.ropeSin.dispose();
        this.ropeInvFreq.dispose();
    }
}
