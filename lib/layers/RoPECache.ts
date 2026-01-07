import { cos, div, keep, pow, range, reciprocal, scalar, sin, Tensor, tidy } from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops';
import { GPTConfig } from '../models/config';

export default class RoPECache {
    public readonly rotaryDim: number;
    private ropeBase: number;
    private ropeInvFreq: Tensor;
    private ropeCos: Tensor | null = null; // [cacheLen, rotaryDim/2]
    private ropeSin: Tensor | null = null; // [cacheLen, rotaryDim/2]
    private ropeNegSin: Tensor | null = null; // [cacheLen, rotaryDim/2]
    private ropeCacheLen = 0;

    constructor(private readonly config: GPTConfig) {
        const headDim = this.config.nEmbed / this.config.nHead;
        this.rotaryDim = headDim;
        if (this.rotaryDim % 2 !== 0) {
            throw new Error('rotaryDim must be even');
        }
        this.ropeBase = 10000; // Could be a little smaller for shorter sequences
        const i = range(0, this.rotaryDim, 2, 'float32'); // even indices
        const exponent = div(i, scalar(this.rotaryDim, 'float32')); // i/rotaryDim
        const basePow = pow(scalar(this.ropeBase, 'float32'), exponent);
        this.ropeInvFreq = reciprocal(basePow); // [rotaryDim/2]

        exponent.dispose();
        basePow.dispose();
        i.dispose();

        if (this.config.useRope === false) {
            this.ropeCos = null;
            this.ropeSin = null;
            this.ropeNegSin = null;
            this.ropeCacheLen = 0;
        } else {
            tidy(() => {
                this.ensureRopeCache(this.config.blockSize * 4);
            });
        }
    }

    public ensureRopeCache(needed: number) {
        tidy(() => {
            if (needed <= this.ropeCacheLen) return;
            if (this.ropeCos) this.ropeCos.dispose();
            if (this.ropeSin) this.ropeSin.dispose();
            const nextSize = Math.max(needed, this.ropeCacheLen + this.config.blockSize * 4);
            const positions = range(0, nextSize, 1, 'float32').expandDims(1); // [L,1]
            const freqs = positions.mul(this.ropeInvFreq.expandDims(0)); // [L, rd/2]
            this.ropeCos = keep(cos(freqs).expandDims(-1)); // [L, rd/2, 1]
            this.ropeSin = keep(sin(freqs).expandDims(-1)); // [L, rd/2, 1]
            this.ropeNegSin = keep(this.ropeSin.neg()); // [L, rd/2, 1]
            this.ropeCacheLen = nextSize;
        });
    }

    public getCos(): Tensor | null {
        return this.ropeCos;
    }

    public getSin(): Tensor | null {
        return this.ropeSin;
    }

    public getNegSin(): Tensor | null {
        return this.ropeNegSin;
    }

    public dispose() {
        if (this.ropeCos) this.ropeCos.dispose();
        if (this.ropeSin) this.ropeSin.dispose();
        this.ropeInvFreq.dispose();
    }
}
