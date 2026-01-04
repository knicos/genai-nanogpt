import { createReduceInfo, reduce, ReduceProgram } from './utils/reductions';
import {
    backend_util,
    KernelConfig,
    NamedAttrMap,
    NamedTensorInfoMap,
    registerKernel,
    sum,
    Tensor,
    TensorInfo,
    util,
} from '@tensorflow/tfjs-core';
import { isPackedTensor } from '@base/utilities/packed';
import { transpose16 } from '../transpose16';
import WebGPUBackendPatch from '@base/patches/webgpu_backend';
import createDeviceInformation, { DeviceInformation } from './utils/deviceInfo';

class SumProgram16 extends ReduceProgram {
    shaderKey = 'sum16';

    constructor(deviceInfo: DeviceInformation, reduceInfo: backend_util.ReduceInfo, packed: boolean) {
        super(
            deviceInfo,
            reduceInfo,
            {
                reductionOp: 'sum',
                elementwise: false,
            },
            packed
        );
        if (packed) {
            this.shaderKey += '_packed';
        }
    }
}

function sum16GPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { axis, keepDims } = args.attrs as { axis?: number | number[]; keepDims?: boolean };
    const backend = args.backend as WebGPUBackendPatch;
    const toDispose: Tensor[] = [];

    const deviceInfo = createDeviceInformation(backend);

    const packed = isPackedTensor(x);

    if (!packed) {
        return sum(x, axis, keepDims);
    }

    const origAxes = util.parseAxisParam(axis ?? -1, x.shape);
    let axes = origAxes;
    const permutedAxes = backend_util.getAxesPermutation(axes, x.shape.length);

    let input = x;
    if (permutedAxes != null) {
        input = transpose16(x, permutedAxes);
        axes = backend_util.getInnerMostAxes(axes.length, input.shape.length);
        toDispose.push(input);
    }

    const reduceInfo = createReduceInfo([input], -1);
    const program = new SumProgram16(deviceInfo, reduceInfo, packed);

    const result = reduce(program, [input], backend);
    toDispose.forEach((t) => t.dispose());
    return result;
}

const kernelConfig: KernelConfig = {
    kernelName: 'Sum16',
    backendName: 'webgpu',
    kernelFunc: sum16GPU,
};

registerKernel(kernelConfig);
