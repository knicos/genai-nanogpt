import RoPECache from '@base/layers/RoPECache';
import { GPGPUProgram, MathBackendWebGL } from '@tensorflow/tfjs-backend-webgl';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    Tensor,
} from '@tensorflow/tfjs-core';

class RopeProgram implements GPGPUProgram {
    variableNames = ['x', 'sin', 'cos'];
    outputShape: number[];
    userCode: string;
    // enableShapeUniforms = true;
    customUniforms = [{ name: 'pastLen', type: 'int' as UniformType }];

    constructor(batch: number, heads: number, T: number, C: number) {
        this.outputShape = [batch, heads, T, C];

        this.userCode = `
        void main() {
            ivec4 coords = getOutputCoords(); // [b, h, t, d]
            int b = coords.x;
            int h = coords.y;
            int t = coords.z;
            int d = coords.w;

            int rotaryDim = ${C};

            float outVal = 0.0;

            if (d < rotaryDim) {
                int pairIdx = d / 2;
                float cos = getCos(t + pastLen, pairIdx, 0);
                float sin = getSin(t + pastLen, pairIdx, 0);

                if (d % 2 == 0) {
                    // even index
                    float even = getX(b, h, t, d);
                    float odd = getX(b, h, t, d + 1);
                    outVal = even * cos - odd * sin;
                } else {
                    // odd index
                    float even = getX(b, h, t, d - 1);
                    float odd = getX(b, h, t, d);
                    outVal = even * sin + odd * cos;
                }
            } else {
                // pass through for non-rotary dims
                outVal = getX(b, h, t, d);
            }

            setOutput(outVal);
        }
        `;
    }
}

function ropeGPU(args: { inputs: NamedTensorInfoMap; backend: unknown; attrs?: NamedAttrMap }): TensorInfo {
    const { x } = args.inputs as { x: Tensor };
    const { pastLen, ropeCache, negSin } = args.attrs as unknown as {
        pastLen: number;
        ropeCache: RoPECache;
        negSin: boolean;
    };
    const sin = negSin ? ropeCache.getNegSin()! : ropeCache.getSin()!;
    const cos = ropeCache.getCos()!;
    const backend = args.backend as MathBackendWebGL;

    const batchSize = x.shape[0];
    const heads = x.shape[1]!;
    const seqLength = x.shape[2]!;
    const C = x.shape[3]!;

    const program = new RopeProgram(batchSize, heads, seqLength, C);
    return backend.runWebGLProgram(program, [x, sin, cos], 'float32', [[pastLen]]);
}

const kernelConfig: KernelConfig = {
    kernelName: 'Rope',
    backendName: 'webgl',
    kernelFunc: ropeGPU,
};

registerKernel(kernelConfig);
