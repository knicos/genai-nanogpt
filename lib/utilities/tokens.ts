// Utility for "flat" slicing over fixed-size Uint16 shards (last shard may be shorter).
export function sliceUint16Shards(
    shards: readonly Uint16Array[],
    start: number,
    endExclusive: number,
    shardSize = shards[0]?.length ?? 0
): Uint16Array {
    if (shards.length === 0) return new Uint16Array(0);
    if (!Number.isInteger(start) || !Number.isInteger(endExclusive)) {
        throw new RangeError('start/endExclusive must be integers');
    }
    if (start < 0 || endExclusive < start) {
        throw new RangeError(`Invalid range [${start}, ${endExclusive})`);
    }
    if (shardSize <= 0) {
        throw new RangeError('Invalid shardSize');
    }

    const last = shards[shards.length - 1];
    const totalLength = (shards.length - 1) * shardSize + last.length;

    if (endExclusive > totalLength) {
        throw new RangeError(`Slice end ${endExclusive} is out of bounds for total length ${totalLength}`);
    }
    if (start === endExclusive) return new Uint16Array(0);

    const out = new Uint16Array(endExclusive - start);
    let writeOffset = 0;
    let cursor = start;

    while (cursor < endExclusive) {
        const shardIndex = Math.floor(cursor / shardSize);
        const shardOffset = cursor - shardIndex * shardSize;
        const shard = shards[shardIndex];

        const take = Math.min(endExclusive - cursor, shard.length - shardOffset);
        try {
            out.set(shard.subarray(shardOffset, shardOffset + take), writeOffset);
        } catch (error) {
            console.error(
                `Error slicing shards: shardIndex=${shardIndex}, shardOffset=${shardOffset}, take=${take}, writeOffset=${writeOffset}`
            );

            // Check all shards have correct length
            shards.forEach((s, i) => {
                if (s.length !== shardSize && i !== shards.length - 1) {
                    console.error(`Shard ${i} has length ${s.length}, expected ${shardSize}`);
                }
            });

            throw error;
        }

        cursor += take;
        writeOffset += take;
    }

    return out;
}

// Utility for "flat" slicing over fixed-size Uint16 shards (last shard may be shorter).
export function sliceUint8Shards(
    shards: readonly Uint8Array[],
    start: number,
    endExclusive: number,
    shardSize = shards[0]?.length ?? 0
): Uint8Array {
    if (shards.length === 0) return new Uint8Array(0);
    if (!Number.isInteger(start) || !Number.isInteger(endExclusive)) {
        throw new RangeError('start/endExclusive must be integers');
    }
    if (start < 0 || endExclusive < start) {
        throw new RangeError(`Invalid range [${start}, ${endExclusive})`);
    }
    if (shardSize <= 0) {
        throw new RangeError('Invalid shardSize');
    }

    const last = shards[shards.length - 1];
    const totalLength = (shards.length - 1) * shardSize + last.length;

    if (endExclusive > totalLength) {
        throw new RangeError(`Slice end ${endExclusive} is out of bounds for total length ${totalLength}`);
    }
    if (start === endExclusive) return new Uint8Array(0);

    const out = new Uint8Array(endExclusive - start);
    let writeOffset = 0;
    let cursor = start;

    while (cursor < endExclusive) {
        const shardIndex = Math.floor(cursor / shardSize);
        const shardOffset = cursor - shardIndex * shardSize;
        const shard = shards[shardIndex];

        const take = Math.min(endExclusive - cursor, shard.length - shardOffset);
        out.set(shard.subarray(shardOffset, shardOffset + take), writeOffset);

        cursor += take;
        writeOffset += take;
    }

    return out;
}
