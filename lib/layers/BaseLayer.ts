import MemoryProfiler from '@base/utilities/profile';

export default abstract class BaseLayer {
    protected _profiler?: MemoryProfiler;

    getProfiler(): MemoryProfiler | undefined {
        return this._profiler;
    }

    setProfiler(value: MemoryProfiler | undefined): void {
        this._profiler = value;
    }

    public startMemory() {
        this._profiler?.startMemory();
    }

    public endMemory(label: string) {
        this._profiler?.endMemory(label);
    }
}
