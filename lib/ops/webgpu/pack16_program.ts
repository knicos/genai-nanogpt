import type { WebGPUProgram } from '@tensorflow/tfjs-backend-webgpu';
import { computeDispatch, flatDispatchLayout } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_util';
import { getMainHeaderString as main } from '@tensorflow/tfjs-backend-webgpu/dist/webgpu_program';

export default class PackProgram implements WebGPUProgram {
    outputShape: number[];
    shaderKey = 'Pack16';
    dispatchLayout: { x: number[] };
    dispatch: [number, number, number];
    workgroupSize: [number, number, number] = [64, 1, 1];
    variableNames = ['x'];
    uniforms?: string;
    size = true;
    outputComponent = 4;
    scaling = false;
    padding = 0;

    constructor(outShape: number[], padding = 0) {
        if (outShape[outShape.length - 1] % 2 !== 0 && padding === 0) {
            throw new Error('Last dimension of output shape must be even to use Pack16.');
        }
        if (padding % 4 !== 0) {
            throw new Error('Padding must be a multiple of 4 to use Pack16.');
        }
        this.outputShape = [...outShape.slice(0, -1), outShape[outShape.length - 1]];

        if (padding > 0) {
            this.shaderKey += `_Padded${padding}`;
            this.padding = padding;

            // Update output shape to ensure it is modulo padding
            for (let i = this.outputShape.length - 2; i < this.outputShape.length; i++) {
                if (this.outputShape[i] % this.padding !== 0) {
                    this.outputShape[i] += this.padding - (this.outputShape[i] % this.padding);
                }
            }

            this.outputComponent = 1;
        }

        this.outputShape[this.outputShape.length - 1] /= 2;

        if (this.outputShape[this.outputShape.length - 1] % this.outputComponent !== 0) {
            this.outputComponent = 1;
        }

        this.dispatchLayout = flatDispatchLayout(this.outputShape);
        this.dispatch = computeDispatch(this.dispatchLayout, this.outputShape, this.workgroupSize, [
            this.outputComponent,
            1,
            1,
        ]);
    }

    useScaling() {
        this.shaderKey += '_Scaled';
        this.uniforms = 'scaling : f32,';
        this.scaling = true;
    }

    getUserCode(): string {
        if (this.padding > 0 && this.outputComponent === 1) {
            const coordRank = this.outputShape.length;

            return `
                ${main('index')} {
                    if (index < uniforms.size) {
                        var coords = getCoordsFromIndex(index);
                        coords[${coordRank} - 1] = coords[${coordRank} - 1] * 2;
                        let row = coords[${coordRank} - 2];
                        let col = coords[${coordRank} - 1];
                        let width = uniforms.xShape[${coordRank} - 1];
                        let height = uniforms.xShape[${coordRank} - 2];
                    
                        var value1 = 0.0f;
                        if (col < width && row < height) {
                            let baseInputIndex = getIndexFromCoords${coordRank}D(coords, uniforms.xShape);
                            value1 = x[baseInputIndex] ${this.scaling ? '* uniforms.scaling' : ''};
                        }
                        var value2 = 0.0f;
                        if (col + 1 < width && row < height) {
                            coords[${coordRank} - 1] = coords[${coordRank} - 1] + 1;
                            let baseInputIndex = getIndexFromCoords${coordRank}D(coords, uniforms.xShape);
                            value2 = x[baseInputIndex] ${this.scaling ? '* uniforms.scaling' : ''};
                        }
                        let packed = i32(pack2x16float(vec2<f32>(value1, value2)));
                        result[index] = packed;
                    }
                }`;
        }
        if (this.outputComponent === 1) {
            return `
                ${main('index')} {
                    if (index < uniforms.size) {
                        let baseInputIndex =  index * 2;
                        let x1 = x[baseInputIndex] ${this.scaling ? '* uniforms.scaling' : ''};
                        let x2 = x[baseInputIndex + 1] ${this.scaling ? '* uniforms.scaling' : ''};
                        let packed = i32(pack2x16float(vec2<f32>(x1, x2)));
                        result[index] = packed;
                    }
                }`;
        }
        return `
                ${main('index')} {
                    if (index < uniforms.size) {
                        let baseInputIndex =  index * 2;
                        let x1 = x[baseInputIndex] ${this.scaling ? '* uniforms.scaling' : ''};
                        let x2 = x[baseInputIndex + 1] ${this.scaling ? '* uniforms.scaling' : ''};
                        let packed = vec4<i32>(
                            i32(pack2x16float(vec2<f32>(x1.x, x1.y))),
                            i32(pack2x16float(vec2<f32>(x1.z, x1.w))),
                            i32(pack2x16float(vec2<f32>(x2.x, x2.y))),
                            i32(pack2x16float(vec2<f32>(x2.z, x2.w)))
                        );
                        result[index] = packed;
                    }
                }`;
    }
}
