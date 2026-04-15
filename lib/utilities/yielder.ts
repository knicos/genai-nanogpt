export async function yieldIfNeeded(lastYield: number, cb?: (vocab: number) => void, param?: number): Promise<number> {
    const now = performance.now();
    if (now - lastYield > 40) {
        await new Promise(requestAnimationFrame);
        if (cb) {
            cb(param ?? 0);
        }
        return now;
    }
    return lastYield;
}
