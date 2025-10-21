import { Tensor, tidy } from '@tensorflow/tfjs-core';

export default async function performanceTest(fn: () => Tensor, iterations: number = 10): Promise<number> {
    for (let i = 0; i < 10; i++) {
        const t = fn(); // Warm-up
        await t.data();
        t.dispose();
    }

    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
        const result = tidy(fn);
        if (i === iterations - 1) {
            await result.data();
            result.dispose();
        } else {
            result.dispose();
        }
    }

    const end = performance.now();
    return (end - start) / iterations;
}
