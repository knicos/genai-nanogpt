export function arraysClose(a: unknown, b: unknown, epsilon = 1e-5) {
    if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; ++i) {
            if (!arraysClose(a[i], b[i], epsilon)) return false;
        }
        return true;
    } else if (typeof a === 'number' && typeof b === 'number') {
        if (a === -Infinity && b === -Infinity) return true;
        return Math.abs(a - b) < epsilon;
    } else {
        return false;
    }
}
