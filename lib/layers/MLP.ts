import { randomNormal, Tensor, tidy, variable } from '@tensorflow/tfjs-core';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { GPTConfig } from '@base/main';
import { matMul16 } from '@base/ops/matMul16';
import { reshape16 } from '@base/ops/reshape16';

export interface MLPConfig {
    activation?: 'gelu' | 'relu2';
    hiddenFactor?: number;
}

// Multi-layer perceptron
export default class MLP extends BaseLayer {
    private index: number;
    private hiddenUnits: number;
    private MLPHIDDEN: string;
    private MLPOUT: string;
    private mlpConfig: MLPConfig;

    constructor(index: number, config: GPTConfig, mlpConfig: MLPConfig, parent?: BaseLayer) {
        super(config, parent);
        this.index = index;
        this.mlpConfig = mlpConfig;
        this.hiddenUnits = (mlpConfig.hiddenFactor ?? config.mlpFactor) * config.nEmbed;

        this.MLPHIDDEN = `block_${this.index}_mlpHidden`;
        this.MLPOUT = `block_${this.index}_mlpOut`;

        this.addVariable(this.MLPHIDDEN);
        this.addVariable(this.MLPOUT);
    }

    protected override build() {
        if (this.hasVariable(this.MLPHIDDEN) === false) {
            this.setVariable(
                this.MLPHIDDEN,
                variable(randomNormal([this.config.nEmbed, this.hiddenUnits], 0, 0.02), true, this.MLPHIDDEN)
            );
        }
        if (this.hasVariable(this.MLPOUT) === false) {
            this.setVariable(
                this.MLPOUT,
                variable(
                    randomNormal([this.hiddenUnits, this.config.nEmbed], 0, 0.02 / Math.sqrt(2 * this.config.nLayer)),
                    true,
                    this.MLPOUT
                )
            );
        }
    }

    forward(_: ForwardAttributes, x: Tensor): Tensor {
        return tidy(() => {
            this.startMemory();
            const [B, T, C] = x.shape;
            const x2d = reshape16(x, [B! * T!, C!]); // (B*T, C)

            const h = matMul16(x2d, this.getVariable(this.MLPHIDDEN), false, false, {
                activation: this.mlpConfig.activation ?? 'gelu',
            }); // (B*T, hidden)

            const out2d = matMul16(h, this.getVariable(this.MLPOUT)); // (B*T, C)
            h.dispose();

            const projected = reshape16(out2d, [B!, T!, C!]);
            this.endMemory('MLP');
            return projected;
        });
    }
}
