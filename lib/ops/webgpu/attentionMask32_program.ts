import { WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';

export default class AttentionMaskProgram32 implements WebGPUProgram {
    variableNames = ['q', 'k'];
    outputShape: number[];

    shaderKey = 'AttentionMask';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    uniforms = 'divisor: f32, pastLen: i32, inf: f32';
    workgroupSize: [number, number, number] = [64, 1, 1];
    size = true;
    hs: number;
    nh: number;
    T1: number;
    T2: number;

    constructor(batch: number, nh: number, T1: number, T2: number, hs: number) {
        this.shaderKey = `AttentionMask_${hs}`;
        this.outputShape = [batch, nh, T1, T2];
        this.hs = hs;
        this.nh = nh;
        this.T1 = T1;
        this.T2 = T2;

        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize);

        if (hs % 4 !== 0) {
            throw new Error('Head size must be a multiple of 4 for AttentionMaskProgram');
        }
    }

    getUserCode(): string {
        const userCode = `
            ${main('index')} {
                
                let coords = getCoordsFromIndex(index);
                let b = coords[0];
                let h = coords[1];
                let t1 = coords[2];
                let t2 = coords[3];

                if (index < uniforms.size) {
                    if (t2 > t1 + uniforms.pastLen) {
                        setOutputAtIndex(index, uniforms.inf);
                        return;
                    }

                    let q0 = getIndexFromCoords4D(vec4<i32>(b, h, t1, 0), uniforms.qShape);
                    let k0 = getIndexFromCoords4D(vec4<i32>(b, h, t2, 0), uniforms.kShape);
                    
                    var sum: f32 = 0.0;
                    for (var i: i32 = 0; i < ${this.hs}; i = i + 4) {
                        let qv = vec4<f32>(q[q0 + i], q[q0 + i + 1], q[q0 + i + 2], q[q0 + i + 3]);
                        let kv = vec4<f32>(k[k0 + i], k[k0 + i + 1], k[k0 + i + 2], k[k0 + i + 3]);
                        sum = sum + dot(qv, kv);
                    }
                    let scaled = sum * uniforms.divisor;
                    setOutputAtIndex(index, scaled);
                }
            }
        `;
        return userCode;
    }
}
