import RoPECache from '@base/layers/RoPECache';
import { GPGPUProgram, MathBackendWebGL, Tensor, engine } from '@tensorflow/tfjs';
import { UniformType } from '@tensorflow/tfjs-backend-webgl/dist/shader_compiler';
import {
    registerKernel,
    KernelConfig,
    TensorInfo,
    NamedTensorInfoMap,
    NamedAttrMap,
    registerGradient,
    GradConfig,
    range,
    gather,
    stack,
    concat,
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
    const { x, sin, cos } = args.inputs as { x: Tensor; sin: Tensor; cos: Tensor };
    const { pastLen } = args.attrs as { pastLen: number };

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

function applyRoPE(sinCache: Tensor, cosCache: Tensor, rotaryDim: number, q: Tensor, pastLen: number): Tensor {
    const hs = q.shape[3]!;
    const rd = rotaryDim;
    if (rd > hs) return q;

    const Tcur = q.shape[2]!;
    //const endPos = pastLen + Tcur;
    //cache.ensureRopeCache(endPos);

    const half = rd / 2;

    // Use rank-4 tensors for WebGL compatibility (avoid 5D broadcasting)
    const cos = cosCache.slice([pastLen, 0, 0], [Tcur, half, 1]).reshape([1, 1, Tcur, half]);
    const sin = sinCache.slice([pastLen, 0, 0], [Tcur, half, 1]).reshape([1, 1, Tcur, half]);

    const B = q.shape[0]!;
    const nh = q.shape[1]!;

    const evenIdx = range(0, rd, 2, 'int32');
    const oddIdx = range(1, rd, 2, 'int32');

    const rotate = (x: Tensor) => {
        const rotPart = x.slice([0, 0, 0, 0], [B, nh, Tcur, rd]);
        const restPart = rd < hs ? x.slice([0, 0, 0, rd], [B, nh, Tcur, hs - rd]) : null;

        const even = gather(rotPart, evenIdx, 3); // [B, nh, Tcur, half]
        const odd = gather(rotPart, oddIdx, 3); // [B, nh, Tcur, half]

        const evenCost = even.mul(cos);
        const oddSin = odd.mul(sin);
        const evenRot = evenCost.sub(oddSin);
        const oddCost = odd.mul(cos);
        const evenSin = even.mul(sin);
        const oddRot = oddCost.add(evenSin);

        even.dispose();
        odd.dispose();
        cos.dispose();
        sin.dispose();
        evenCost.dispose();
        oddSin.dispose();
        oddCost.dispose();
        evenSin.dispose();

        // Interleave (even', odd') -> last dim size rd, without elementwise ops on rank-5
        const stacked = stack([evenRot, oddRot], -1); // [B, nh, Tcur, half, 2]
        evenRot.dispose();
        oddRot.dispose();
        const rotated = stacked.reshape([B, nh, Tcur, rd]); // [B, nh, Tcur, rd]
        stacked.dispose();

        return restPart ? concat([rotated, restPart], 3) : rotated;
    };

    const qR = rotate(q);

    evenIdx.dispose();
    oddIdx.dispose();

    return qR;
}

// CPU fallback implementation
export function ropeCPU(args: { inputs: NamedTensorInfoMap; attrs?: NamedAttrMap }): TensorInfo {
    const { x, sin, cos } = args.inputs as { x: Tensor; sin: Tensor; cos: Tensor };
    const { pastLen } = args.attrs as { pastLen: number };

    const rotaryDim = x.shape[3]!;
    return applyRoPE(sin, cos, rotaryDim, x, pastLen);
}

const cpuKernelConfig: KernelConfig = {
    kernelName: 'Rope',
    backendName: 'cpu',
    kernelFunc: ropeCPU,
};

registerKernel(cpuKernelConfig);

const tensorflowKernelConfig: KernelConfig = {
    kernelName: 'Rope',
    backendName: 'tensorflow',
    kernelFunc: ropeCPU,
};

registerKernel(tensorflowKernelConfig);

export function rope(x: Tensor, cache: RoPECache, pastLength: number): Tensor {
    cache.ensureRopeCache(x.shape[1]! + pastLength); // x.shape[1] = Tcur
    return engine().runKernel('Rope', { x, sin: cache.getSin()!, cos: cache.getCos()! }, { pastLen: pastLength });
}

const ropeGradConfig: GradConfig = {
    kernelName: 'Rope',
    inputsToSave: ['x', 'sin', 'cos'],
    outputsToSave: [],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        // dy: gradient of output
        // x: input tensor
        // sin, cos: caches

        const [x, sin, cos] = saved as Tensor[];

        // To invert RoPE, apply RoPE with -sin (i.e., swap sin sign)
        // This is mathematically equivalent to applying the inverse rotation

        // Negate sin cache
        const negSin = sin.neg();

        // Use the same applyRoPE logic, but with negated sin
        const rotaryDim = x.shape[3]!;
        const pastLen = 0; // You may need to pass the correct pastLen if used

        const gradInput = applyRoPE(negSin, cos, rotaryDim, dy as Tensor, pastLen);

        negSin.dispose();

        return { x: () => gradInput };
    },
};

registerGradient(ropeGradConfig);
