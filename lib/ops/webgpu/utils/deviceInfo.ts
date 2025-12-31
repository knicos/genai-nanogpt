import WebGPUBackendPatch from '@base/patches/webgpu_backend';

export interface DeviceInformation {
    subgroupsSupported: boolean;
    subgroupMaxSize: number;
    variableSubgroups: boolean;
}

export default function createDeviceInformation(backend: WebGPUBackendPatch): DeviceInformation {
    const hasSubgroupSupport = backend.device.features.has('subgroups');
    return {
        subgroupsSupported: hasSubgroupSupport,
        subgroupMaxSize: backend.subgroupMaxSize,
        variableSubgroups: backend.subgroupMinSize !== backend.subgroupMaxSize && hasSubgroupSupport,
    };
}
