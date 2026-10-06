import { randomNormal, Tensor, tidy, variable } from '@tensorflow/tfjs-core';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { GPTConfig } from '@base/models/config';
import { matMul16 } from '@base/ops/matMul16';
import { reshape16 } from '@base/ops/reshape16';
import { dropout16 } from '@base/ops/dropout16';
import { mul16 } from '@base/ops/mul16';

export interface MLPConfig {
    activation?: 'gelu' | 'relu2' | 'swiglu';
    hiddenFactor?: number;
    hiddenSize?: number;
}

// Round size to nearest multiple of a given number
export function swigluHidden(C: number, factor = 8 / 3, multipleOf = 32): number {
    const h = Math.round(factor * C);
    return Math.max(multipleOf, Math.round(h / multipleOf) * multipleOf);
}

// Multi-layer perceptron
export default class MLP extends BaseLayer {
    private index: number;
    private hiddenUnits: number;
    private MLPHIDDEN: string;
    private MLPOUT: string;
    private MLPGATE: string;
    private MLPUP: string;
    private mlpConfig: MLPConfig;

    constructor(index: number, config: GPTConfig, mlpConfig: MLPConfig, parent?: BaseLayer) {
        super(config, parent);
        this.index = index;
        this.mlpConfig = mlpConfig;
        this.hiddenUnits =
            mlpConfig.hiddenSize ?? swigluHidden(config.nEmbed, mlpConfig.hiddenFactor ?? config.mlpFactor);

        this.MLPHIDDEN = `block_${this.index}_mlpHidden`;
        this.MLPOUT = `block_${this.index}_mlpOut`;

        this.addVariable(this.MLPOUT);
        this.MLPGATE = `block_${this.index}_mlpGate`;
        this.MLPUP = `block_${this.index}_mlpUp`;
        if (this.mlpConfig.activation === 'swiglu') {
            this.addVariable(this.MLPGATE);
            this.addVariable(this.MLPUP);
        } else {
            this.addVariable(this.MLPHIDDEN);
        }
    }

    protected override build() {
        if (this.mlpConfig.activation === 'swiglu') {
            if (this.hasVariable(this.MLPGATE) === false) {
                this.setVariable(
                    this.MLPGATE,
                    variable(randomNormal([this.config.nEmbed, this.hiddenUnits], 0, 0.02), true, this.MLPGATE)
                );
            }
            if (this.hasVariable(this.MLPUP) === false) {
                this.setVariable(
                    this.MLPUP,
                    variable(randomNormal([this.config.nEmbed, this.hiddenUnits], 0, 0.02), true, this.MLPUP)
                );
            }
        } else {
            if (this.hasVariable(this.MLPHIDDEN) === false) {
                this.setVariable(
                    this.MLPHIDDEN,
                    variable(randomNormal([this.config.nEmbed, this.hiddenUnits], 0, 0.02), true, this.MLPHIDDEN)
                );
            }
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

    forward(attr: ForwardAttributes, x: Tensor): Tensor {
        return tidy(() => {
            this.startMemory();
            const [B, T, C] = x.shape;
            const x2d = reshape16(x, [B! * T!, C!]); // (B*T, C)

            let h: Tensor;

            if (this.mlpConfig.activation === 'swiglu') {
                const gate = matMul16(x2d, this.getVariable(this.MLPGATE), false, false, {
                    activation: 'silu', // silu(x) = x * sigmoid(x)
                }); // (B*T, hidden)
                const up = matMul16(x2d, this.getVariable(this.MLPUP)); // (B*T, hidden)
                h = mul16(gate, up); // (B*T, hidden)
                gate.dispose();
                up.dispose();
            } else {
                h = matMul16(x2d, this.getVariable(this.MLPHIDDEN), false, false, {
                    activation: this.mlpConfig.activation ?? 'gelu',
                }); // (B*T, hidden)
            }

            const out2d = matMul16(h, this.getVariable(this.MLPOUT)); // (B*T, C)
            h.dispose();
            const projected = reshape16(out2d, [B!, T!, C!]);
            this.endMemory('MLP');

            if (attr.dropout && attr.dropout > 0) {
                return dropout16(projected, attr.dropout);
            }

            return projected;
        });
    }
}
