export function arraysClose(a: unknown, b: unknown) {
    let maxError = 0.0;
    if ((Array.isArray(a) || a instanceof Float32Array) && (Array.isArray(b) || b instanceof Float32Array)) {
        if (a.length !== b.length) return Number.POSITIVE_INFINITY;
        for (let i = 0; i < a.length; ++i) {
            maxError = Math.max(maxError, arraysClose(a[i], b[i]));
        }
        return maxError;
    } else if (typeof a === 'number' && typeof b === 'number') {
        if (isNaN(a) && isNaN(b)) {
            return 0.0;
        }
        if (!isFinite(a) || !isFinite(b)) {
            return a === b ? 0.0 : Number.POSITIVE_INFINITY;
        }
        const aClose = Math.abs(a - b);
        maxError = Math.max(maxError, aClose);
        return maxError;
    } else {
        return Number.POSITIVE_INFINITY;
    }
}

export function arraysClosePercentile(a: unknown, b: unknown, percentile = 0.99): number {
    const errors: number[] = [];

    function walk(x: unknown, y: unknown): boolean {
        if ((Array.isArray(x) || x instanceof Float32Array) && (Array.isArray(y) || y instanceof Float32Array)) {
            if (x.length !== y.length) return false;
            for (let i = 0; i < x.length; i++) {
                if (!walk(x[i], y[i])) return false;
            }
            return true;
        }

        if (typeof x === 'number' && typeof y === 'number') {
            if (Number.isNaN(x) && Number.isNaN(y)) {
                errors.push(0);
                return true;
            }
            if (!Number.isFinite(x) || !Number.isFinite(y)) {
                return x === y;
            }
            errors.push(Math.abs(x - y));
            return true;
        }

        return false;
    }

    if (!walk(a, b)) return Number.POSITIVE_INFINITY;
    if (errors.length === 0) return Number.POSITIVE_INFINITY;

    errors.sort((m, n) => m - n);
    const idx = Math.min(errors.length - 1, Math.floor(percentile * errors.length));
    return errors[idx];
}
