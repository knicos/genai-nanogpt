import type TF from '@tensorflow/tfjs';
import { GPTConfig } from '../config';

export default class RoPECache {
    private rotaryDim: number;
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

        if (this.config.useRope === false) {
            this.ropeCos = null;
            this.ropeSin = null;
            this.ropeCacheLen = 0;
        } else {
            this.ensureRopeCache(this.config.blockSize * 4);
        }
    }

    private ensureRopeCache(needed: number) {
        if (needed <= this.ropeCacheLen) return;
        if (this.ropeCos) this.ropeCos.dispose();
        if (this.ropeSin) this.ropeSin.dispose();
        const positions = this.tf.range(0, needed, 1, 'float32').expandDims(1); // [L,1]
        const freqs = positions.mul(this.ropeInvFreq.expandDims(0)); // [L, rd/2]
        this.ropeCos = this.tf.keep(this.tf.cos(freqs).expandDims(-1)); // [L, rd/2, 1]
        this.ropeSin = this.tf.keep(this.tf.sin(freqs).expandDims(-1)); // [L, rd/2, 1]
        this.ropeCacheLen = needed;
    }

    public applyRoPE(q: TF.Tensor, k: TF.Tensor, pastLen: number): [TF.Tensor, TF.Tensor] {
        const hs = q.shape[3]!;
        const rd = this.rotaryDim;
        if (rd > hs) return [q, k];

        const Tcur = q.shape[2]!;
        const endPos = pastLen + Tcur;
        this.ensureRopeCache(endPos);

        const half = rd / 2;

        const cos = (this.ropeCos as TF.Tensor).slice([pastLen, 0, 0], [Tcur, half, 1]);
        const sin = (this.ropeSin as TF.Tensor).slice([pastLen, 0, 0], [Tcur, half, 1]);
        const cosB = cos.reshape([1, 1, Tcur, half, 1]);
        const sinB = sin.reshape([1, 1, Tcur, half, 1]);

        const both = this.tf.concat([q, k], 0);
        const B2 = both.shape[0]!;
        const nh = both.shape[1]!;

        const rotPart = both.slice([0, 0, 0, 0], [B2, nh, Tcur, rd]);
        const restPart = rd < hs ? both.slice([0, 0, 0, rd], [B2, nh, Tcur, hs - rd]) : null;

        const pairs = rotPart.reshape([B2, nh, Tcur, half, 2]);
        const even = pairs.slice([0, 0, 0, 0, 0], [B2, nh, Tcur, half, 1]);
        const odd = pairs.slice([0, 0, 0, 0, 1], [B2, nh, Tcur, half, 1]);

        const evenRot = even.mul(cosB).sub(odd.mul(sinB));
        const oddRot = odd.mul(cosB).add(even.mul(sinB));

        const rotated = this.tf.concat([evenRot, oddRot], -1).reshape([B2, nh, Tcur, rd]);

        const merged = restPart ? this.tf.concat([rotated, restPart], 3) : rotated;
        const B = B2 / 2;
        const qR = merged.slice([0, 0, 0, 0], [B, nh, Tcur, hs]);
        const kR = merged.slice([B, 0, 0, 0], [B, nh, Tcur, hs]);

        return [qR, kR];
    }

    public dispose() {
        if (this.ropeCos) this.ropeCos.dispose();
        if (this.ropeSin) this.ropeSin.dispose();
        this.ropeInvFreq.dispose();
    }
}
