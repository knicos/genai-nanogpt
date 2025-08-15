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

        // Use rank-4 tensors for WebGL compatibility (avoid 5D broadcasting)
        const cos = (this.ropeCos as TF.Tensor).slice([pastLen, 0, 0], [Tcur, half, 1]).reshape([1, 1, Tcur, half]);
        const sin = (this.ropeSin as TF.Tensor).slice([pastLen, 0, 0], [Tcur, half, 1]).reshape([1, 1, Tcur, half]);

        const B = q.shape[0]!;
        const nh = q.shape[1]!;

        const evenIdx = this.tf.range(0, rd, 2, 'int32');
        const oddIdx = this.tf.range(1, rd, 2, 'int32');

        const rotate = (x: TF.Tensor) => {
            const rotPart = x.slice([0, 0, 0, 0], [B, nh, Tcur, rd]);
            const restPart = rd < hs ? x.slice([0, 0, 0, rd], [B, nh, Tcur, hs - rd]) : null;

            const even = this.tf.gather(rotPart, evenIdx, 3); // [B, nh, Tcur, half]
            const odd = this.tf.gather(rotPart, oddIdx, 3); // [B, nh, Tcur, half]

            const evenRot = even.mul(cos).sub(odd.mul(sin));
            const oddRot = odd.mul(cos).add(even.mul(sin));

            // Interleave (even', odd') -> last dim size rd, without elementwise ops on rank-5
            const stacked = this.tf.stack([evenRot, oddRot], -1); // [B, nh, Tcur, half, 2]
            const rotated = stacked.reshape([B, nh, Tcur, rd]); // [B, nh, Tcur, rd]

            return restPart ? this.tf.concat([rotated, restPart], 3) : rotated;
        };

        const qR = rotate(q);
        const kR = rotate(k);

        evenIdx.dispose();
        oddIdx.dispose();

        return [qR, kR];
    }

    public dispose() {
        if (this.ropeCos) this.ropeCos.dispose();
        if (this.ropeSin) this.ropeSin.dispose();
        this.ropeInvFreq.dispose();
    }
}
