import { getBackend, ready, setBackend } from '@tensorflow/tfjs-core';

export async function selectBackend(backendName: 'cpu' | 'webgl' | 'webgpu'): Promise<void> {
    if (getBackend() !== backendName) {
        if (backendName === 'webgpu') {
            await import(`@tensorflow/tfjs-backend-webgpu`);
            await import(`./ops/webgpu/index`);
        }
        await setBackend(backendName);
        await ready();
        console.log(`Backend set to ${backendName}`);
    }
}
