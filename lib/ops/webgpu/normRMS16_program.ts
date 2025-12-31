import { backend_util } from '@tensorflow/tfjs-core';
import { ReduceProgram } from './utils/reductions';
import { DeviceInformation } from './utils/deviceInfo';

export default class RMSProgram16 extends ReduceProgram {
    constructor(deviceInfo: DeviceInformation, reduceInfo: backend_util.ReduceInfo) {
        super(deviceInfo, reduceInfo, { reductionOp: 'mean', elementwise: true }, true);
        this.shaderKey = 'RMSNorm16';
        this.variableNames.push('gamma');
        this.variableComponents = [1, 1];
    }

    override getPreprocessSnippet(): string {
        return 'candidate = candidate * candidate;';
    }

    override getPostprocessSnippet(): string {
        return 'bestValue = inverseSqrt(bestValue + 1e-8);';
    }

    override getWriteSnippet(): string {
        return `
            let X = unpack2x16float(u32(x[offset + k]));
            let gamma = unpack2x16float(u32(gamma[k]));
            let normalized = X * bestValue;
            let outVal = normalized * gamma;
            result[offset + k] = i32(pack2x16float(outVal));
        `;
    }
}
