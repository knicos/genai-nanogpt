import { tensor } from '@tensorflow/tfjs';
import { describe, it } from 'vitest';
import { load_safetensors, save_safetensors } from './safetensors';

describe('safetensors', () => {
    it('should save and load tensors correctly', async ({ expect }) => {
        const tensorData = {
            a: tensor([1, 2, 3], [3], 'float32'),
            b: tensor(
                [
                    [1, 2],
                    [3, 4],
                ],
                [2, 2],
                'float32'
            ),
        };

        const buffer = await save_safetensors(tensorData);
        const loadedTensors = await load_safetensors(buffer);

        console.log('F32 Buffer length', buffer.byteLength);

        expect(loadedTensors).toHaveProperty('a');
        expect(loadedTensors).toHaveProperty('b');

        expect(loadedTensors['a'].shape).toEqual([3]);
        expect(loadedTensors['b'].shape).toEqual([2, 2]);

        expect(Array.from(await loadedTensors['a'].data())).toEqual([1, 2, 3]);
        expect(Array.from(await loadedTensors['b'].data())).toEqual([1, 2, 3, 4]);
    });

    it('should support float16 quantization', async ({ expect }) => {
        const tensorData = {
            a: tensor([1, 2, 3], [3], 'float32'),
            b: tensor(
                [
                    [1, 2],
                    [3, 4],
                ],
                [2, 2],
                'float32'
            ),
        };

        const buffer = await save_safetensors(tensorData, 'F16');
        const buffer32 = await save_safetensors(tensorData, 'none');
        const loadedTensors = await load_safetensors(buffer);

        const countFloats = tensorData.a.size + tensorData.b.size;
        const byteSizeDifference = countFloats * 2;

        expect(buffer32.byteLength - buffer.byteLength).toEqual(byteSizeDifference);

        expect(loadedTensors).toHaveProperty('a');
        expect(loadedTensors).toHaveProperty('b');

        expect(loadedTensors['a'].shape).toEqual([3]);
        expect(loadedTensors['b'].shape).toEqual([2, 2]);

        expect(Array.from(await loadedTensors['a'].data())).toEqual([1, 2, 3]);
        expect(Array.from(await loadedTensors['b'].data())).toEqual([1, 2, 3, 4]);
    });

    it('should support int32', async ({ expect }) => {
        const tensorData = {
            a: tensor([1, 2, 3], [3], 'int32'),
            b: tensor(
                [
                    [1, 2],
                    [3, 4],
                ],
                [2, 2],
                'int32'
            ),
        };

        const buffer = await save_safetensors(tensorData);
        const loadedTensors = await load_safetensors(buffer);

        expect(loadedTensors).toHaveProperty('a');
        expect(loadedTensors).toHaveProperty('b');

        expect(loadedTensors['a'].shape).toEqual([3]);
        expect(loadedTensors['b'].shape).toEqual([2, 2]);
        expect(loadedTensors['b'].dtype).toEqual('int32');

        expect(Array.from(await loadedTensors['a'].data())).toEqual([1, 2, 3]);
        expect(Array.from(await loadedTensors['b'].data())).toEqual([1, 2, 3, 4]);
    });

    it('supports empty tensors', async ({ expect }) => {
        const tensorData = {
            a: tensor([], [0], 'float32'),
            b: tensor([], [0, 2], 'float32'),
        };

        const buffer = await save_safetensors(tensorData);
        const loadedTensors = await load_safetensors(buffer);

        expect(loadedTensors).toHaveProperty('a');
        expect(loadedTensors).toHaveProperty('b');

        expect(loadedTensors['a'].shape).toEqual([0]);
        expect(loadedTensors['b'].shape).toEqual([0, 2]);

        expect(Array.from(await loadedTensors['a'].data())).toEqual([]);
        expect(Array.from(await loadedTensors['b'].data())).toEqual([]);
    });
});
