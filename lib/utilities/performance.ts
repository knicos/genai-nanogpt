import { Tensor, tidy } from '@tensorflow/tfjs-core';

export default async function performanceTest(
    fn: () => Tensor,
    iterations: number = 10,
    allowPromise: boolean = false
): Promise<number> {
    for (let i = 0; i < 100; i++) {
        const t = allowPromise ? await fn() : tidy(fn); // Warm-up
        if (i === 99) await t.data();
        t.dispose();
    }

    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
        const result = allowPromise ? await fn() : tidy(fn);
        if (i === iterations - 1) {
            await result.data();
        }
        result.dispose();
    }

    const end = performance.now();
    return (end - start) / iterations;
}
