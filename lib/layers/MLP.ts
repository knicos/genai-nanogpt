import {
    customGrad,
    dropout,
    engine,
    grads,
    matMul,
    randomNormal,
    reshape,
    Tensor,
    tidy,
    variable,
    Variable,
} from '@tensorflow/tfjs-core';
import BaseLayer, { GPTLayerConfig } from './BaseLayer';
import { matMulGelu } from '@base/ops/matMulGelu';

// Multi-layer perceptron
export default class MLP extends BaseLayer {
    private cFc: Variable | null = null;
    private cProj: Variable | null = null;
    private index: number;
    private _trainable: boolean = true;
    private hiddenUnits: number;

    constructor(index: number, config: GPTLayerConfig) {
        super(config);
        this.index = index;
        this.hiddenUnits = config.gpt.mlpFactor * config.gpt.nEmbed;
    }

    private build() {
        if (this.cFc === null) {
            this.cFc = variable(
                randomNormal([this.config.gpt.nEmbed, this.hiddenUnits], 0, 0.02),
                true
                //`block_${this.index}_attn_cAttn_kernel`
            );
        }
        if (this.cProj === null) {
            this.cProj = variable(
                randomNormal(
                    [this.hiddenUnits, this.config.gpt.nEmbed],
                    0,
                    0.02 / Math.sqrt(2 * this.config.gpt.nLayer)
                ),
                true
                //`block_${this.index}_attn_cProj_kernel`
            );
        }
    }

    get variables(): Variable[] {
        return [this.cFc!, this.cProj!];
    }

    get trainable(): boolean {
        return this._trainable;
    }

    set trainable(value: boolean) {
        this._trainable = value;
        if (this.cFc) this.cFc.trainable = value;
        if (this.cProj) this.cProj.trainable = value;
    }

    saveWeights(map: Map<string, Tensor[]>): void {
        map.set(`block_${this.index}_mlpHidden`, this.cFc ? [this.cFc.clone()] : []);
        map.set(`block_${this.index}_mlpOut`, this.cProj ? [this.cProj.clone()] : []);
    }

    loadWeights(weights: Map<string, Tensor[]>): void {
        const projWeight = weights.get(`block_${this.index}_mlpOut`)?.[0];
        const fcWeight = weights.get(`block_${this.index}_mlpHidden`)?.[0];
        if (!projWeight || !fcWeight) {
            throw new Error(`Weights for block ${this.index} not found`);
        }
        if (this.cFc) {
            this.cFc.assign(fcWeight);
        } else {
            this.cFc = variable(fcWeight, true); //, `block_${this.index}_attn_cHidden_kernel`);
        }
        if (this.cProj) {
            this.cProj.assign(projWeight);
        } else {
            this.cProj = variable(projWeight, true); //, `block_${this.index}_attn_cProj_kernel`);
        }
    }

    forward(x: Tensor): Tensor {
        return tidy(() => {
            this.startMemory();
            const [B, T, C] = x.shape;
            const x2d = reshape(x, [B! * T!, C!]); // (B*T, C)
            //const h = matMul(x2d, this.cFc!); // (B*T, hidden)
            //const act = gelu(h);
            const h = matMulGelu(x2d, this.cFc!); // (B*T, hidden)
            const out2d = matMul(h, this.cProj!); // (B*T, C)
            h.dispose();
            const projected = reshape(out2d, [B!, T!, C!]);
            this.endMemory('MLP');
            return projected;
        });
    }

    call(x: Tensor, training = false): Tensor {
        this.build();

        if (training && this.config.layerConfig.checkpointMLP) {
            const cpMLP = customGrad(
                // @ts-expect-error Invalid params
                (x: Tensor, fcVar: Tensor, projVar: Tensor, save: (tensors: Tensor[]) => void) => {
                    const output = this.forward(x);

                    save([x]);
                    const gradFunc = (dy: Tensor, saved: Tensor[]) => {
                        const [xSaved] = saved;

                        // Hack to allow nested grads calls
                        const savedTape = engine().state.activeTape;
                        engine().state.activeTape = [];

                        // Recompute forward pass
                        // We need to pass fcVar and projVar to keep them in scope for the backward pass
                        const g = grads((x: Tensor, fcVar: Tensor, projVar: Tensor) => {
                            void fcVar;
                            void projVar;
                            const output = this.forward(x);
                            return output;
                        })([xSaved, fcVar, projVar], dy);

                        // Restore tape
                        engine().state.activeTape = savedTape;

                        return g;
                    };
                    return { value: output, gradFunc };
                }
            );

            const output = cpMLP(x, this.cFc!, this.cProj!);
            if (this.config.gpt.dropout > 0) {
                const dOut = dropout(output, this.config.gpt.dropout);
                output.dispose();
                return dOut;
            }
            return output;
        } else {
            const output = this.forward(x);
            if (training && this.config.gpt.dropout > 0) {
                const dOut = dropout(output, this.config.gpt.dropout);
                output.dispose();
                return dOut;
            }
            return output;
        }
    }

    dispose() {
        this.cFc?.dispose();
        this.cProj?.dispose();
    }
}
