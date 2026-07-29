import { describe, it } from 'vitest';
import { manualFloat16Array, toFloat16Array } from './float16';

describe('float16', () => {
    it('correctly implements a manual float16 conversion', async ({ expect }) => {
        const float32Values = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0]);
        const manual = manualFloat16Array(float32Values);
        const converted = toFloat16Array(float32Values);
        expect(manual).toEqual(converted);
    });
});
