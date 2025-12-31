import { DataType, env, TensorInfo, util } from '@tensorflow/tfjs-core';
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu';
import { compileProgram, WebGPUProgram } from './webgpu_program';
import { makeShaderKey } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';

type ProgramUniform = Array<{ type: string; data: number[] }>;

// Reshape dispatch, not to exceed device limits.
const reshapeDispatch = (device: GPUDevice, program: WebGPUProgram): [number, number, number] => {
    const MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE = device.limits.maxComputeWorkgroupsPerDimension;
    const layout = program.dispatchLayout;
    const dispatch = program.dispatch;
    if (dispatch.every((d) => d <= MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE)) {
        return dispatch;
    }

    util.assert(
        dispatch[0] > MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE && layout.y === undefined && layout.z === undefined,
        () => 'Dispatch size exceeds WebGPU limits in Y or Z dimension.'
    );

    let dispatchAverage = Math.ceil(Math.sqrt(dispatch[0]));
    if (dispatchAverage > MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE) {
        dispatchAverage = Math.ceil(Math.cbrt(dispatch[0]));
        util.assert(
            dispatchAverage <= MAX_COMPUTE_PER_DIMENSION_DISPATCH_SIZE,
            () => 'Total dispatch size exceeds WebGPU maximum.'
        );
        return [dispatchAverage, dispatchAverage, dispatchAverage];
    } else {
        return [dispatchAverage, dispatchAverage, 1];
    }
};

interface ExtendedAdapterInfo extends GPUAdapterInfo {
    subgroupMaxSize?: number;
    subgroupMinSize?: number;
}

export default class WebGPUBackendPatch extends WebGPUBackend {
    public readonly subgroupMaxSize: number;
    public readonly subgroupMinSize: number;

    constructor(device: GPUDevice, adapterInfo?: ExtendedAdapterInfo) {
        super(device, adapterInfo);
        this.subgroupMaxSize = adapterInfo?.subgroupMaxSize ?? 0;
        this.subgroupMinSize = adapterInfo?.subgroupMinSize ?? 0;
    }

    override runWebGPUProgram(
        program: WebGPUProgram,
        inputs: TensorInfo[],
        outputDtype: DataType,
        programDefinedUniform?: ProgramUniform,
        output?: TensorInfo
    ): TensorInfo {
        if (!output) {
            output = this.makeTensorInfo(program.outputShape, outputDtype);
        }
        if (util.sizeFromShape(output.shape) === 0) {
            // Short-circuit the computation since the result is empty (has 0 in its
            // shape).
            this.tensorMap.get(output.dataId).values = util.getTypedArrayFromDType(output.dtype as 'float32', 0);
            return output;
        }
        this.uploadToGPU(output.dataId);
        program.dispatch = reshapeDispatch(this.device, program);

        const inputsData = inputs.map((input: TensorInfo, i: number) => {
            if (input.dtype === 'complex64') {
                throw new Error(
                    `GPGPUProgram does not support complex64 input. For complex64 ` +
                        `dtypes, please separate the program into real and imaginary ` +
                        `parts.`
                );
            }
            this.uploadToGPU(input.dataId);

            return {
                // Returning dtype from tensorMap because it reflects dtype
                // of underlying buffer, rather than abstract dtype.
                dtype: this.tensorMap.get(input.dataId).dtype,
                shape: input.shape,
                name: program.variableNames[i],
            };
        });

        program.shaderKey = makeShaderKey(program, inputsData, output);

        const parallelCompilation = env().getBool('WEBGPU_ENGINE_COMPILE_ONLY');
        if (!(program.shaderKey in this['pipelineCache'])) {
            this['pipelineCache'][program.shaderKey] = compileProgram(
                this.device,
                program,
                inputsData,
                output,
                parallelCompilation
            );
        }
        program.pipeline = this['pipelineCache'][program.shaderKey];

        if (!parallelCompilation) {
            this['recordAndSubmit'](program, output, inputs, programDefinedUniform);
        }
        return output;
    }
}
