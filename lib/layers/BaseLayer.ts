import { GPTConfig } from '@base/config';
import MemoryProfiler from '@base/utilities/profile';
import RoPECache from './RoPECache';

export interface LayerConfig {
    checkpointAttention?: boolean; // Whether to use gradient checkpointing for attention layers
    checkpointMLP?: boolean; // Whether to use gradient checkpointing for MLP layers
    profiler?: MemoryProfiler;
    ropeCache?: RoPECache;
}

export interface GPTLayerConfig {
    gpt: GPTConfig;
    layerConfig: LayerConfig;
}

export default abstract class BaseLayer {
    public readonly config: GPTLayerConfig;

    constructor(config: GPTLayerConfig) {
        this.config = config;
    }

    public getProfiler(): MemoryProfiler | undefined {
        return this.config.layerConfig.profiler;
    }

    public startMemory() {
        this.config.layerConfig.profiler?.startMemory();
    }

    public endMemory(label: string) {
        this.config.layerConfig.profiler?.endMemory(label);
    }
}
