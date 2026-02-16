import { ones, Tensor, tidy, variable } from '@tensorflow/tfjs-core';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { normRMS } from '@base/ops/normRMS';
import { GPTConfig } from '@base/main';

export interface RMSNormConfig {
    useGamma?: boolean;
}

export default class RMSNorm extends BaseLayer {
    private GAMMA: string;
    private rmsConfig: RMSNormConfig;

    constructor(config: GPTConfig, rmsConfig: RMSNormConfig, name = '', parent?: BaseLayer) {
        super(config, parent);
        this.GAMMA = name;
        this.rmsConfig = rmsConfig;

        if (this.rmsConfig.useGamma ?? true) {
            this.addVariable(this.GAMMA, variable(ones([config.nEmbed]), true, this.GAMMA, 'float32'));
        }
    }

    forward(_: ForwardAttributes, x: Tensor): Tensor {
        return tidy(() => {
            this.startMemory();
            const result = normRMS(x, (this.rmsConfig.useGamma ?? true) ? this.getVariable(this.GAMMA) : undefined);
            this.endMemory('RMSNorm');
            return result;
        });
    }
}
