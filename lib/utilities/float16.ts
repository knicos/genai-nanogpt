export function toFloat16Array(float32arr: Float32Array): Uint16Array {
    if (typeof Float16Array !== 'undefined') {
        const f16 = new Float16Array(float32arr);
        return new Uint16Array(f16.buffer);
    }
    return manualFloat16Array(float32arr);
}

export function loadFloat16FromBuffer(buffer: ArrayBuffer, byteOffset: number, length: number): Float32Array {
    if (typeof Float16Array !== 'undefined') {
        // Native: view directly, then copy out as Float32Array of real numbers
        const view = new Float16Array(buffer, byteOffset, length);
        return new Float32Array(view); // widens each value to a JS number
    }
    return manualLoadFloat16(buffer, byteOffset, length);
}

function manualLoadFloat16(buffer: ArrayBuffer, byteOffset: number, length: number): Float32Array {
    const bits = new Uint16Array(buffer, byteOffset, length);
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        out[i] = float16BitsToNumber(bits[i]);
    }
    return out;
}

function float16BitsToNumber(bits: number): number {
    const sign = bits & 0x8000 ? -1 : 1;
    const exp = (bits >>> 10) & 0x1f;
    const mant = bits & 0x3ff;

    if (exp === 0) {
        // Subnormal or zero
        return sign * mant * Math.pow(2, -24);
    } else if (exp === 0x1f) {
        // Inf or NaN
        return mant ? NaN : sign * Infinity;
    } else {
        // Normalized
        return sign * (1 + mant / 1024) * Math.pow(2, exp - 15);
    }
}

// Fallback: stores IEEE 754 half-precision bits in a Uint16Array
export function manualFloat16Array(float32arr: Float32Array): Uint16Array {
    const out = new Uint16Array(float32arr.length);
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);

    for (let i = 0; i < float32arr.length; i++) {
        f32[0] = float32arr[i];
        const x = u32[0];

        const sign = (x >>> 16) & 0x8000;
        const exp = ((x >>> 23) & 0xff) - 127 + 15;
        let mant = x & 0x7fffff;

        if (((x >>> 23) & 0xff) === 0xff) {
            // Inf or NaN
            out[i] = sign | 0x7c00 | (mant ? 0x200 : 0);
        } else if (exp >= 0x1f) {
            // Overflow -> Inf
            out[i] = sign | 0x7c00;
        } else if (exp <= 0) {
            // Subnormal or zero
            if (exp < -10) {
                out[i] = sign; // too small -> zero
            } else {
                mant |= 0x800000;
                const shift = 14 - exp;
                let half = mant >>> shift;
                // round to nearest even
                if ((mant >>> (shift - 1)) & 1) {
                    if (mant & ((1 << (shift - 1)) - 1) || half & 1) half++;
                }
                out[i] = sign | half;
            }
        } else {
            // Normalized
            let half = (exp << 10) | (mant >>> 13);
            // round to nearest even
            if (mant & 0x1000) {
                if (mant & 0xfff || half & 1) {
                    half++;
                    if ((half & 0x7c00) === 0x7c00) half = sign | 0x7c00; // overflowed to Inf
                }
            }
            out[i] = sign | half;
        }
    }

    return out;
}
