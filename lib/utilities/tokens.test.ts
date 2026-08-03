import { describe, it } from 'vitest';
import { sliceUint16Shards } from './tokens';

function u16(values: number[]): Uint16Array {
    return new Uint16Array(values);
}

describe('sliceUint16Shards', () => {
    it('returns an empty array when shards are empty', ({ expect }) => {
        const result = sliceUint16Shards([], 0, 0);
        expect(result).toBeInstanceOf(Uint16Array);
        expect(Array.from(result)).toEqual([]);
    });

    it('returns an empty array for an empty range', ({ expect }) => {
        const shards = [u16([10, 11, 12, 13]), u16([14, 15])];
        const result = sliceUint16Shards(shards, 3, 3);
        expect(Array.from(result)).toEqual([]);
    });

    it('slices correctly within a single shard', ({ expect }) => {
        const shards = [u16([1, 2, 3, 4]), u16([5, 6, 7])];
        const result = sliceUint16Shards(shards, 1, 3);
        expect(Array.from(result)).toEqual([2, 3]);
    });

    it('slices correctly across a shard boundary', ({ expect }) => {
        const shards = [u16([1, 2, 3, 4]), u16([5, 6, 7, 8]), u16([9])];
        const result = sliceUint16Shards(shards, 2, 6);
        expect(Array.from(result)).toEqual([3, 4, 5, 6]);
    });

    it('slices correctly across multiple shard boundaries including a short last shard', ({ expect }) => {
        const shards = [u16([1, 2, 3, 4]), u16([5, 6, 7, 8]), u16([9, 10])];
        const result = sliceUint16Shards(shards, 1, 9);
        expect(Array.from(result)).toEqual([2, 3, 4, 5, 6, 7, 8, 9]);
    });

    it('can return the entire flattened sequence', ({ expect }) => {
        const shards = [u16([100, 101, 102]), u16([103, 104, 105]), u16([106])];
        const result = sliceUint16Shards(shards, 0, 7, 3);
        expect(Array.from(result)).toEqual([100, 101, 102, 103, 104, 105, 106]);
    });

    it('returns a copy rather than a view into source shards', ({ expect }) => {
        const shards = [u16([1, 2, 3, 4]), u16([5, 6, 7, 8])];
        const result = sliceUint16Shards(shards, 2, 6);

        result[0] = 999;

        expect(shards[0][2]).toBe(3);
        expect(Array.from(result)).toEqual([999, 4, 5, 6]);
    });

    it('throws when start is negative', ({ expect }) => {
        const shards = [u16([1, 2, 3])];
        expect(() => sliceUint16Shards(shards, -1, 1)).toThrow(RangeError);
    });

    it('throws when end is less than start', ({ expect }) => {
        const shards = [u16([1, 2, 3])];
        expect(() => sliceUint16Shards(shards, 2, 1)).toThrow(RangeError);
    });

    it('throws when start or end are not integers', ({ expect }) => {
        const shards = [u16([1, 2, 3, 4])];
        expect(() => sliceUint16Shards(shards, 0.5, 2)).toThrow(RangeError);
        expect(() => sliceUint16Shards(shards, 0, 2.2)).toThrow(RangeError);
    });

    it('throws when end is out of bounds', ({ expect }) => {
        const shards = [u16([1, 2, 3, 4]), u16([5, 6])];
        expect(() => sliceUint16Shards(shards, 0, 7)).toThrow(RangeError);
    });

    it('throws when shardSize is invalid', ({ expect }) => {
        const shards = [u16([1, 2, 3])];
        expect(() => sliceUint16Shards(shards, 0, 1, 0)).toThrow(RangeError);
        expect(() => sliceUint16Shards(shards, 0, 1, -4)).toThrow(RangeError);
    });
});
