import type TF from '@tensorflow/tfjs';

export interface ITensorSpec {
    shape: number[];
    min?: number;
    scale?: number;
}

export interface IWeightManifest {
    spec: ITensorSpec[];
    data: Float32Array;
}

function float32Concat(arrays: Float32Array[]): Float32Array {
    const totalLength = arrays.reduce((sum, arr) => sum + arr.length, 0);
    const result = new Float32Array(totalLength);
    let offset = 0;
    for (const arr of arrays) {
        result.set(arr, offset);
        offset += arr.length;
    }
    return result;
}

export async function exportWeights(weights: TF.Tensor[]): Promise<IWeightManifest> {
    const manifest: IWeightManifest = {
        spec: [],
        data: new Float32Array(),
    };

    const dataArrays: Float32Array[] = [];

    for (const tensor of weights) {
        if (!tensor || !Array.isArray(tensor.shape) || tensor.shape.length === 0) {
            console.warn(`Skipping weight with invalid shape:`, tensor);
            continue;
        }
        const minT = tensor.min();
        const maxT = tensor.max();
        const min = (await minT.data())[0];
        const scale = (await maxT.data())[0] - min;
        manifest.spec.push({
            shape: tensor.shape,
            min,
            scale,
        });

        minT.dispose();
        maxT.dispose();
        const data = await tensor.data<'float32'>();
        //tensor.dispose();
        dataArrays.push(data);
    }
    manifest.data = float32Concat(dataArrays);
    return manifest;
}

export async function importWeights(manifest: IWeightManifest, tf: typeof TF): Promise<TF.Tensor[]> {
    const weights: TF.Tensor[] = [];
    let offset = 0;

    for (const spec of manifest.spec) {
        const size = spec.shape.reduce((a, b) => a * b, 1);
        const data = manifest.data.slice(offset, offset + size);
        offset += size;

        const tensor = tf.tensor(data, spec.shape, 'float32');
        /*if (spec.min !== undefined && spec.scale !== undefined) {
            tensor.sub(spec.min).div(spec.scale);
        }*/
        weights.push(tensor);
    }

    return weights;
}
