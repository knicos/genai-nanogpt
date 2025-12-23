import { dropout, randomNormal, reshape, Tensor, tidy, variable } from '@tensorflow/tfjs-core';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { GPTConfig } from '@base/main';
import { matMul16, matMul16Gelu } from '@base/ops/matMul16';
import { pack16 } from '@base/ops/pack16';
import { unpack16 } from '@base/ops/unpack16';

// Multi-layer perceptron
export default class MLP extends BaseLayer {
    private index: number;
    private hiddenUnits: number;
    private MLPHIDDEN: string;
    private MLPOUT: string;

    constructor(index: number, config: GPTConfig, parent?: BaseLayer) {
        super(config, parent);
        this.index = index;
        this.hiddenUnits = config.mlpFactor * config.nEmbed;

        this.MLPHIDDEN = `block_${this.index}_mlpHidden`;
        this.MLPOUT = `block_${this.index}_mlpOut`;

        this.addVariable(this.MLPHIDDEN);
        this.addVariable(this.MLPOUT);
    }

    protected override build() {
        if (this.hasVariable(this.MLPHIDDEN) === false) {
            this.setVariable(
                this.MLPHIDDEN,
                variable(
                    randomNormal([this.config.nEmbed, this.hiddenUnits], 0, 0.02),
                    true,
                    `block_${this.index}_mlpHidden_kernel`
                )
            );
        }
        if (this.hasVariable(this.MLPOUT) === false) {
            this.setVariable(
                this.MLPOUT,
                variable(
                    randomNormal([this.hiddenUnits, this.config.nEmbed], 0, 0.02 / Math.sqrt(2 * this.config.nLayer)),
                    true,
                    `block_${this.index}_mlpOut_kernel`
                )
            );
        }
    }

    forward(attr: ForwardAttributes, x: Tensor): Tensor {
        return tidy(() => {
            this.startMemory();
            const [B, T, C] = x.shape;
            const x2d = reshape(x, [B! * T!, C!]); // (B*T, C)

            const packedx2d = attr.mixedPrecision ? pack16(x2d) : x2d;

            if (attr.mixedPrecision) {
                x2d.dispose();
            }

            const h = matMul16Gelu(packedx2d, this.getVariable(this.MLPHIDDEN)); // (B*T, hidden)

            if (attr.mixedPrecision) {
                packedx2d.dispose();
            }

            const out2d = matMul16(h, this.getVariable(this.MLPOUT)); // (B*T, C)
            h.dispose();

            const unpackedOut2d = attr.mixedPrecision ? unpack16(out2d) : out2d;
            if (attr.mixedPrecision) {
                out2d.dispose();
            }

            const projected = reshape(unpackedOut2d, [B!, T!, C!]);
            this.endMemory('MLP');
            return projected;
        });
    }

    protected override dropout(x: Tensor): Tensor {
        if (this.config.dropout > 0) {
            const dOut = dropout(x, this.config.dropout);
            x.dispose();
            return dOut;
        }
        return x;
    }
}
