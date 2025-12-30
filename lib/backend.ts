import { getBackend, ready, setBackend } from '@tensorflow/tfjs-core';

export async function selectBackend(
    backendName: 'cpu' | 'webgl' | 'webgpu',
    gpuPreference?: 'low-power' | 'high-performance'
): Promise<void> {
    if (getBackend() !== backendName) {
        if (backendName === 'webgpu') {
            const { registerWebGPUBackend } = await import(`./patches/webgpu_base`);
            registerWebGPUBackend(gpuPreference ?? 'high-performance');
            await import(`@tensorflow/tfjs-backend-webgpu`);
            await import(`./ops/webgpu/index`);
        }
        await setBackend(backendName);
        await ready();
        console.log(`Backend set to ${backendName}`);
    }
}
