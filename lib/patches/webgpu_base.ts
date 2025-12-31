/**
 * @license
 * Copyright 2022 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// Nick: Modified to control GPU selection and enable optional features.

import { registerBackend } from '@tensorflow/tfjs-core';
import WebGPUBackend from './webgpu_backend';

export interface GPUOptions {
    powerPreference?: 'low-power' | 'high-performance';
    disableSubgroups?: boolean;
}

export function registerWebGPUBackend(options?: GPUOptions): void {
    registerBackend(
        'webgpu',
        async () => {
            const gpuDescriptor: GPURequestAdapterOptions = {
                powerPreference: options?.powerPreference ?? 'high-performance',
            };

            console.log('Using custom WebGPU backend with power preference:', gpuDescriptor.powerPreference);

            const adapter = (await navigator.gpu.requestAdapter(gpuDescriptor))!;
            const deviceDescriptor: GPUDeviceDescriptor = {};

            const requiredFeatures = [];
            if (adapter.features.has('timestamp-query')) {
                requiredFeatures.push('timestamp-query');
            }
            if (adapter.features.has('bgra8unorm-storage')) {
                requiredFeatures.push(['bgra8unorm-storage']);
            }
            if (!options?.disableSubgroups && adapter.features.has('subgroups')) {
                requiredFeatures.push('subgroups');
            }
            deviceDescriptor.requiredFeatures = requiredFeatures as Iterable<GPUFeatureName>;

            const adapterLimits = adapter.limits;
            deviceDescriptor.requiredLimits = {
                maxComputeWorkgroupStorageSize: adapterLimits.maxComputeWorkgroupStorageSize,
                maxComputeWorkgroupsPerDimension: adapterLimits.maxComputeWorkgroupsPerDimension,
                maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
                maxBufferSize: adapterLimits.maxBufferSize,
                maxComputeWorkgroupSizeX: adapterLimits.maxComputeWorkgroupSizeX,
                maxComputeInvocationsPerWorkgroup: adapterLimits.maxComputeInvocationsPerWorkgroup,
            };

            const device: GPUDevice = await adapter.requestDevice(deviceDescriptor);
            const adapterInfo =
                'info' in adapter
                    ? adapter.info
                    : 'requestAdapterInfo' in adapter
                    ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
                      await (adapter as any).requestAdapterInfo()
                    : undefined;
            return new WebGPUBackend(device, adapterInfo);
        },
        3 /*priority*/
    );
}
