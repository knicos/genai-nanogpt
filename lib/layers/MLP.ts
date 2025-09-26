import { dropout, matMul, randomNormal, reshape, Tensor, tidy, variable } from '@tensorflow/tfjs-core';
import BaseLayer, { ForwardAttributes, GPTLayerConfig } from './BaseLayer';
import { matMulGelu } from '@base/ops/matMulGelu';

// Multi-layer perceptron
export default class MLP extends BaseLayer {
    private index: number;
    private hiddenUnits: number;
    private MLPHIDDEN: string;
    private MLPOUT: string;

    constructor(index: number, config: GPTLayerConfig, parent?: BaseLayer) {
        super(config, parent);
        this.index = index;
        this.hiddenUnits = config.gpt.mlpFactor * config.gpt.nEmbed;

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
                    randomNormal([this.config.gpt.nEmbed, this.hiddenUnits], 0, 0.02),
                    true
                    //`block_${this.index}_attn_cAttn_kernel`
                )
            );
        }
        if (this.hasVariable(this.MLPOUT) === false) {
            this.setVariable(
                this.MLPOUT,
                variable(
                    randomNormal(
                        [this.hiddenUnits, this.config.gpt.nEmbed],
                        0,
                        0.02 / Math.sqrt(2 * this.config.gpt.nLayer)
                    ),
                    true
                    //`block_${this.index}_attn_cProj_kernel`
                )
            );
        }
    }

    forward(_: ForwardAttributes, x: Tensor): Tensor {
        return tidy(() => {
            this.startMemory();
            const [B, T, C] = x.shape;
            const x2d = reshape(x, [B! * T!, C!]); // (B*T, C)
            const h = matMulGelu(x2d, this.getVariable(this.MLPHIDDEN)); // (B*T, hidden)
            const out2d = matMul(h, this.getVariable(this.MLPOUT)); // (B*T, C)
            h.dispose();
            const projected = reshape(out2d, [B!, T!, C!]);
            this.endMemory('MLP');
            return projected;
        });
    }

    protected override dropout(x: Tensor): Tensor {
        if (this.config.gpt.dropout > 0) {
            const dOut = dropout(x, this.config.gpt.dropout);
            x.dispose();
            return dOut;
        }
        return x;
    }
}
