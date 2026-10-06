import { backend_util } from '@tensorflow/tfjs-core';

import { ReduceProgram } from './utils/reductions';
import { DeviceInformation } from './utils/deviceInfo';

export default class RMSProgram32 extends ReduceProgram {
    private hasGamma: boolean;

    constructor(
        deviceInfo: DeviceInformation,
        reduceInfo: backend_util.ReduceInfo,
        hasGamma = false,
        private epsilon = 1e-8
    ) {
        super(deviceInfo, reduceInfo, { reductionOp: 'mean', elementwise: true }, false);
        this.shaderKey = `RMSNorm32_${hasGamma ? 'Gamma' : 'NoGamma'}_${this.epsilon}`;
        if (hasGamma) {
            this.variableNames.push('gamma');
        }
        this.variableComponents = [1, 1];
        this.hasGamma = hasGamma;
    }

    protected override getPreprocessSnippet(): string {
        return 'candidate = candidate * candidate;';
    }

    protected override getPostprocessSnippet(): string {
        return `bestValue = inverseSqrt(bestValue + ${this.epsilon});`;
    }

    protected override getWriteSnippet(): string {
        return `
            let X = f32(x[offset + k]);
            ${this.hasGamma ? `let gamma = gamma[k];` : ''}
            let normalized = X * bestValue;
            let outVal = normalized ${this.hasGamma ? ' * gamma' : ''};
            result[offset + k] = f32(outVal);
        `;
    }
}
