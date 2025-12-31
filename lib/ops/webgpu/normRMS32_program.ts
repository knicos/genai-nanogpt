import { backend_util } from '@tensorflow/tfjs-core';

import { ReduceProgram } from './utils/reductions';
import { DeviceInformation } from './utils/deviceInfo';

export default class RMSProgram32 extends ReduceProgram {
    constructor(deviceInfo: DeviceInformation, reduceInfo: backend_util.ReduceInfo) {
        super(deviceInfo, reduceInfo, { reductionOp: 'mean', elementwise: true }, false);
        this.shaderKey = 'RMSNorm32';
        this.variableNames.push('gamma');
        this.variableComponents = [1, 1];
    }

    protected override getPreprocessSnippet(): string {
        return 'candidate = candidate * candidate;';
    }

    protected override getPostprocessSnippet(): string {
        return 'bestValue = inverseSqrt(bestValue + 1e-8);';
    }

    protected override getWriteSnippet(): string {
        return `
            let X = f32(x[offset + k]);
            let gamma = gamma[k];
            let normalized = X * bestValue;
            let outVal = normalized * gamma;
            result[offset + k] = f32(outVal);
        `;
    }
}
