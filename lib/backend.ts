import { getBackend, ready, setBackend } from '@tensorflow/tfjs-core';
import { GPUOptions } from './patches/webgpu_base';

export async function selectBackend(backendName: 'cpu' | 'webgl' | 'webgpu', options?: GPUOptions): Promise<void> {
    if (getBackend() !== backendName) {
        if (backendName === 'webgpu') {
            const { registerWebGPUBackend } = await import(`./patches/webgpu_base`);
            registerWebGPUBackend(options);
            await import(`@tensorflow/tfjs-backend-webgpu`);
            await import(`./ops/webgpu/index`);
        }
        await setBackend(backendName);
        await ready();
        console.log(`Backend set to ${backendName}`);
    }
}
