import { ones, Tensor, tidy, variable } from '@tensorflow/tfjs-core';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { normRMS } from '@base/ops/normRMS';
import { GPTConfig } from '@base/models/config';
import { isPackedTensor } from '@base/utilities/packed';

export interface RMSNormConfig {
    useGamma?: boolean;
    epsilon?: number;
}

export default class RMSNorm extends BaseLayer {
    private GAMMA: string;
    private rmsConfig: RMSNormConfig;

    constructor(config: GPTConfig, rmsConfig: RMSNormConfig, name = '', parent?: BaseLayer) {
        super(config, parent);
        this.GAMMA = name;
        this.rmsConfig = rmsConfig;

        if (this.rmsConfig.useGamma ?? true) {
            this.addVariable(this.GAMMA);
        }
    }

    protected buildGamma(size: number) {
        if (this.rmsConfig.useGamma ?? true) {
            if (this.hasVariable(this.GAMMA) === false) {
                this.setVariable(this.GAMMA, variable(ones([size]), true, this.GAMMA, 'float32'));
            }
        }
    }

    forward(_: ForwardAttributes, x: Tensor): Tensor {
        return tidy(() => {
            this.startMemory();
            this.buildGamma(isPackedTensor(x) ? x.shape[x.shape.length - 1] * 2 : x.shape[x.shape.length - 1]);
            const result = normRMS(
                x,
                (this.rmsConfig.useGamma ?? true) ? this.getVariable(this.GAMMA) : undefined,
                this.rmsConfig.epsilon
            );
            this.endMemory('RMSNorm');
            return result;
        });
    }
}
