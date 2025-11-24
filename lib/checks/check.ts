import { arraysClose } from '@base/utilities/arrayClose';

interface Result {
    backend: string;
    result: unknown;
    error?: string;
    passed: boolean;
    maxError?: number;
}

export default async function runCheck(
    check: (backend: string) => Promise<unknown>,
    epsilon?: number
): Promise<Result[]> {
    const backends = ['cpu', 'webgl', 'webgpu'];

    const results: Result[] = [];

    for (const backend of backends) {
        try {
            const result = await check(backend);
            results.push({ backend, result, passed: true });
        } catch (error) {
            results.push({ backend, error: (error as Error).message, result: [], passed: false });
        }
    }

    const resultSync = await Promise.all(results);

    const reference = resultSync[0].result;
    for (let i = 1; i < resultSync.length; i++) {
        const current = resultSync[i].result;
        const close = arraysClose(reference, current);
        resultSync[i].passed = close <= (epsilon ?? 1e-6);
        resultSync[i].maxError = close;
    }

    return resultSync;
}
