import { ones, Tensor, tidy, variable } from '@tensorflow/tfjs-core';
import BaseLayer, { ForwardAttributes } from './BaseLayer';
import { normRMS } from '@base/ops/normRMS';
import { GPTConfig } from '@base/main';

export default class RMSNorm extends BaseLayer {
    private GAMMA: string;
    constructor(config: GPTConfig, name = '', parent?: BaseLayer) {
        super(config, parent);
        this.GAMMA = name;
        this.addVariable(this.GAMMA, variable(ones([config.nEmbed]), true, this.GAMMA, 'float32'));
    }

    forward(_: ForwardAttributes, x: Tensor): Tensor {
        return tidy(() => {
            this.startMemory();
            const result = normRMS(x, this.getVariable(this.GAMMA));
            this.endMemory('RMSNorm');
            return result;
        });
    }
}
