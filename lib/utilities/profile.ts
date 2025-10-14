import { memory, MemoryInfo } from '@tensorflow/tfjs-core';

const MB = 1024 * 1024;

export interface ExtendedMemoryInfo extends MemoryInfo {
    numBytesInGPUAllocated?: number;
}

export default class MemoryProfiler {
    private log = new Map<string, number>();
    private maxMemory = 0;
    private maxLabel?: string;
    private lastMemInfo: MemoryInfo[] = [];
    private peakMemory = 0;

    public startMemory() {
        this.lastMemInfo.push(memory());
    }

    public getPeakMemory() {
        return this.peakMemory;
    }

    public getMaxMemory() {
        return this.maxMemory;
    }

    public endMemory(label: string) {
        if (this.lastMemInfo.length === 0) {
            console.warn('MemoryProfiler: endMemory called without matching startMemory');
            return;
        }
        const memoryInfo = memory() as ExtendedMemoryInfo;
        const popped = this.lastMemInfo.pop() as ExtendedMemoryInfo;
        const usedBytes =
            (memoryInfo.numBytesInGPUAllocated ?? memoryInfo.numBytes) -
            (popped?.numBytes ?? popped?.numBytesInGPUAllocated ?? 0);
        this.log.set(label, Math.max(this.log.get(label) || 0, usedBytes));
        if (usedBytes > this.maxMemory) {
            this.maxMemory = usedBytes;
            this.maxLabel = label;
        }

        this.peakMemory = Math.max(this.peakMemory, memoryInfo.numBytesInGPUAllocated ?? memoryInfo.numBytes);
    }

    public printSummary() {
        console.log('Memory Usage Summary:');
        for (const [label, bytes] of this.log.entries()) {
            console.log(`- ${label}: ${(bytes / MB).toFixed(2)} MB`);
        }
        if (this.maxLabel) {
            console.log(`Peak Memory Usage: ${(this.maxMemory / MB).toFixed(2)} MB at "${this.maxLabel}"`);
        }
        console.log(`Overall Peak Memory Usage: ${(this.peakMemory / MB).toFixed(2)} MB`);
    }
}
