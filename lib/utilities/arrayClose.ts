export function arraysClose(a: unknown, b: unknown) {
    let maxError = 0.0;
    if (Array.isArray(a) && Array.isArray(b)) {
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
