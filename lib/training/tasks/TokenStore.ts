type MaybeShard = Uint16Array | null;
type MaybeMask = Uint8Array | null;

const DIR_NAME = 'llm-tokenstore';

interface TokenStoreManifest {
    tokenizerId: string;
    datasetId: string;
    shardSize: number;
    shardCount: number;
    lastShardLength: number;
    maskExistsIndex?: boolean[];
    timestamp: number;
}

export class TokenStore {
    public readonly name: string;
    public readonly tokeniserId: string;
    public readonly datasetId: string;
    private shards: MaybeShard[] = [];
    private masks?: MaybeMask[];
    private shardCount = 0;
    private lastShardLength = -1;
    private _shardSize: number = 8_000 * 1024; // 8 million tokens

    // OPFS
    private dirHandle: FileSystemDirectoryHandle | null = null;
    private opfsAvailable = false;

    // LRU cache bookkeeping (Map preserves insertion order; newest at end)
    private lru = new Map<number, true>();

    // Maximum number of shards to keep in memory
    private maxCachedShards: number;

    constructor(tokeniserId: string, datasetId: string, name?: string, maxCachedShards = 2, shardSize = 8_000 * 1024) {
        this.tokeniserId = tokeniserId;
        this.datasetId = datasetId;
        this.maxCachedShards = Math.max(1, Math.trunc(maxCachedShards));
        this._shardSize = Math.max(1, Math.trunc(shardSize));
        this.name = name ?? DIR_NAME;
    }

    public get shardSize(): number {
        return this._shardSize;
    }

    public hasMask(): boolean {
        return !!this.masks;
    }

    public getShardLength(index: number): number {
        if (index < 0 || index >= this.shardCount) {
            throw new RangeError(`Shard index ${index} out of range`);
        }
        return index === this.shardCount - 1 ? this.lastShardLength : this._shardSize;
    }

    // Helper to get a slice of tokens across shards, handling boundaries
    public async slice(start: number, end: number): Promise<Uint16Array> {
        if (start < 0 || end > this.getTokenCount() || start >= end) {
            throw new RangeError(`Invalid slice range: ${start} to ${end}`);
        }

        const result = new Uint16Array(end - start);
        let offset = 0;

        const startShardIndex = Math.floor(start / this._shardSize);
        const endShardIndex = Math.floor((end - 1) / this._shardSize);

        for (let shardIndex = startShardIndex; shardIndex <= endShardIndex; shardIndex++) {
            const shardStart = shardIndex * this._shardSize;
            const shardEnd = shardStart + this.getShardLength(shardIndex);
            const sliceStart = Math.max(start, shardStart);
            const sliceEnd = Math.min(end, shardEnd);

            if (sliceStart < sliceEnd) {
                const shard = await this.getShard(shardIndex);
                const shardSliceStart = sliceStart - shardStart;
                const shardSliceEnd = sliceEnd - shardStart;
                result.set(shard.subarray(shardSliceStart, shardSliceEnd), offset);
                offset += shardSliceEnd - shardSliceStart;
            }
        }

        return result;
    }

    // Initialize OPFS directory handle if available
    public async init(): Promise<void> {
        try {
            const nav = globalThis.navigator;
            if (!nav?.storage?.getDirectory) {
                this.opfsAvailable = false;
                return;
            }
            // origin-private file system root
            const root = await nav.storage.getDirectory();
            // create/get our directory by id
            this.dirHandle = await root.getDirectoryHandle(this.name, { create: true });
            this.opfsAvailable = true;

            // try to read manifest to restore shardLengths if present
            try {
                const mf = await this.readManifest();

                if (mf && (mf?.tokenizerId !== this.tokeniserId || mf?.datasetId !== this.datasetId)) {
                    // Cleanup any existing shards/masks in OPFS that don't match the current tokenizer/dataset
                    for await (const entry of this.dirHandle.values()) {
                        const name = entry.name;
                        if (name.startsWith('shard-') || name.startsWith('mask-') || name === 'manifest.json') {
                            try {
                                await this.dirHandle.removeEntry(name);
                            } catch {
                                // ignore per-file errors
                            }
                        }
                    }
                } else if (mf) {
                    this.shardCount = mf.shardCount;
                    this._shardSize = mf.shardSize;
                    this.lastShardLength = mf.lastShardLength;
                    // Ensure arrays sizes match shardCount
                    this.shards = new Array(this.shardCount).fill(null);
                    if (mf.maskExistsIndex && Array.isArray(mf.maskExistsIndex)) {
                        this.masks = new Array(this.shardCount).fill(null);
                    }
                }
            } catch {
                // ignore manifest read errors
            }
        } catch {
            this.opfsAvailable = false;
            this.dirHandle = null;
            console.warn('TokenStore: OPFS initialization failed, falling back to memory-only storage');
        }
    }

    private async writeManifest(): Promise<void> {
        if (!this.opfsAvailable || !this.dirHandle) return;
        const manifest: TokenStoreManifest = {
            tokenizerId: this.tokeniserId,
            datasetId: this.datasetId,
            shardSize: this._shardSize,
            shardCount: this.shardCount,
            lastShardLength: this.lastShardLength,
            maskExistsIndex: (this.masks || []).map((m) => !!m),
            timestamp: Date.now(),
        };
        const handle = await this.dirHandle.getFileHandle('manifest.json', { create: true });
        const writable = await handle.createWritable({ keepExistingData: false });
        await writable.write(JSON.stringify(manifest));
        await writable.close();
    }

    private async readManifest(): Promise<TokenStoreManifest | undefined> {
        if (!this.opfsAvailable || !this.dirHandle) return undefined;
        try {
            const fh = await this.dirHandle.getFileHandle('manifest.json');
            const file = await fh.getFile();
            const txt = await file.text();
            return JSON.parse(txt);
        } catch {
            return undefined;
        }
    }

    private shardFilename(index: number) {
        return `shard-${index}.bin`;
    }
    private maskFilename(index: number) {
        return `mask-${index}.bin`;
    }

    private async writeShardToOPFS(index: number, shard: Uint16Array): Promise<void> {
        if (!this.opfsAvailable || !this.dirHandle) return;
        const name = this.shardFilename(index);
        const fh = await this.dirHandle.getFileHandle(name, { create: true });
        const writable = await fh.createWritable();
        await writable.write(shard.buffer as ArrayBuffer);
        await writable.close();
    }

    private async writeMaskToOPFS(index: number, mask: Uint8Array): Promise<void> {
        if (!this.opfsAvailable || !this.dirHandle) return;
        const name = this.maskFilename(index);
        const fh = await this.dirHandle.getFileHandle(name, { create: true });
        const writable = await fh.createWritable();
        await writable.write(mask.buffer as ArrayBuffer);
        await writable.close();
    }

    private async readShardFromOPFS(index: number): Promise<Uint16Array | undefined> {
        if (!this.opfsAvailable || !this.dirHandle) return undefined;
        try {
            const fh = await this.dirHandle.getFileHandle(this.shardFilename(index));
            const file = await fh.getFile();
            const buffer = await file.arrayBuffer();
            return new Uint16Array(buffer);
        } catch {
            return undefined;
        }
    }

    private async readMaskFromOPFS(index: number): Promise<Uint8Array | undefined> {
        if (!this.opfsAvailable || !this.dirHandle) return undefined;
        try {
            const fh = await this.dirHandle.getFileHandle(this.maskFilename(index));
            const file = await fh.getFile();
            const buffer = await file.arrayBuffer();
            return new Uint8Array(buffer);
        } catch {
            return undefined;
        }
    }

    private touchLRU(index: number) {
        // remove and re-insert to mark as most recently used
        this.lru.delete(index);
        this.lru.set(index, true);
    }

    private evictIfNeeded() {
        if (this.maxCachedShards <= 0) return;
        // Evict least recently used unlocked shards until under limit
        while (this.lru.size > this.maxCachedShards) {
            // first key is least recently used
            const it = this.lru.keys();
            const oldest = it.next().value as number | undefined;
            if (oldest === undefined) break;

            // evict
            this.lru.delete(oldest);
            // replace in-memory shard with null to allow GC
            if (this.shards[oldest]) {
                this.shards[oldest] = null;
            }
            if (this.masks?.[oldest]) {
                // we keep masks in OPFS too, unload from memory
                this.masks[oldest] = null;
            }
        }
    }

    public async getShard(index: number): Promise<Uint16Array> {
        if (!Number.isInteger(index) || index < 0 || index >= this.shards.length) {
            throw new RangeError(`Shard index ${index} out of range`);
        }

        // If in memory, return and update LRU
        const inMem = this.shards[index];
        if (inMem) {
            this.touchLRU(index);
            return inMem;
        }

        // Try to load from OPFS
        if (this.opfsAvailable) {
            const shard = await this.readShardFromOPFS(index);
            if (shard) {
                this.shards[index] = shard;
                this.touchLRU(index);
                this.evictIfNeeded();
                return shard;
            }
        }

        throw new Error(`Shard ${index} not available in memory or OPFS`);
    }

    public async getMask(index: number): Promise<Uint8Array | undefined> {
        if (!Number.isInteger(index) || index < 0 || index >= this.shardCount) {
            return undefined;
        }

        if (!this.masks) {
            return undefined;
        }

        const inMem = this.masks[index];
        if (inMem) {
            this.touchLRU(index);
            return inMem;
        }

        if (this.opfsAvailable) {
            const mask = await this.readMaskFromOPFS(index);
            if (mask) {
                this.masks[index] = mask;
                this.touchLRU(index);
                this.evictIfNeeded();
                return mask;
            }
        }
        return undefined;
    }

    public getShardCount(): number {
        return this.shardCount;
    }

    public getTokenCount(): number {
        // Sum of known shard lengths
        if (this.shardCount === 0) return 0;
        return (this.shardCount - 1) * this._shardSize + this.lastShardLength;
    }

    public async finish() {
        await this.writeManifest();
    }

    public appendShard(shard: Uint16Array, mask?: Uint8Array) {
        if (this.lastShardLength >= 0 && this.lastShardLength < this._shardSize) {
            throw new Error('Previous shard was not full');
        }

        const index = this.shards.length;
        this.shards.push(shard);
        if (!this.masks && mask) {
            this.masks = new Array(index).fill(null);
        }
        if (this.masks) {
            this.masks.push(mask ?? null);
        }
        this.shardCount = this.shards.length;
        this.lastShardLength = shard.length;

        // mark as recently used
        this.touchLRU(index);

        // Attempt OPFS writes; resolve when complete. Do not throw to callers on failure.
        try {
            if (this.opfsAvailable && this.dirHandle) {
                this.writeShardToOPFS(index, shard);
                if (mask) this.writeMaskToOPFS(index, mask);
                //this.writeManifest();
            }
        } catch (e) {
            console.error(e);
            // Disable cache eviction if OPFS fails, to avoid losing data in memory
            this.maxCachedShards = Number.MAX_SAFE_INTEGER;
        }

        // Enforce cache size (may evict older unlocked shards)
        this.evictIfNeeded();
    }

    public async dispose(): Promise<void> {
        // Remove all OPFS files for this store (manifest + shards + masks)
        if (this.opfsAvailable && this.dirHandle) {
            try {
                // iterate entries and remove files
                for await (const entry of this.dirHandle.values()) {
                    const name = entry.name;
                    try {
                        await this.dirHandle.removeEntry(name);
                    } catch {
                        // ignore per-file errors
                    }
                }
                // try removing directory from root if possible
                try {
                    const nav = globalThis.navigator;
                    if (nav?.storage?.getDirectory) {
                        const root = await nav.storage.getDirectory();
                        // remove our directory - may not be supported in all implementations
                        if (root.removeEntry) {
                            await root.removeEntry(this.name, { recursive: true });
                        }
                    }
                } catch {
                    // ignore
                }
            } catch {
                // ignore overall cleanup errors
            }
        }

        // Release in-memory references
        this.shards = [];
        this.masks = [];
        this.shardCount = 0;
        this.lastShardLength = -1;
        this.lru.clear();
        this.dirHandle = null;
        this.opfsAvailable = false;
    }

    public async clear(): Promise<void> {
        // Clear in-memory data
        this.shards = [];
        this.masks = [];
        this.shardCount = 0;
        this.lastShardLength = -1;
        this.lru.clear();

        // Clear OPFS data
        if (this.opfsAvailable && this.dirHandle) {
            try {
                for await (const entry of this.dirHandle.values()) {
                    const name = entry.name;
                    try {
                        await this.dirHandle.removeEntry(name);
                    } catch {
                        // ignore per-file errors
                    }
                }
                await this.writeManifest(); // write empty manifest
            } catch {
                // ignore overall cleanup errors
            }
        }
    }
}

const existingStores = new Map<string, TokenStore>();

export function getTokenStore(name: string, tokeniserId: string, datasetId: string): TokenStore | null {
    const existingStore = existingStores.get(name);
    if (existingStore && existingStore.tokeniserId === tokeniserId && existingStore.datasetId === datasetId) {
        return existingStore;
    }
    return null;
}

export async function deleteTokenStore(name: string): Promise<void> {
    const existingStore = existingStores.get(name);
    if (existingStore) {
        await existingStore.dispose();
        existingStores.delete(name);
    } else {
        // Attempt to delete from OPFS if it exists
        try {
            const nav = globalThis.navigator;
            if (nav?.storage?.getDirectory) {
                const root = await nav.storage.getDirectory();
                // remove our directory - may not be supported in all implementations
                if (root.removeEntry) {
                    await root.removeEntry(name, { recursive: true });
                }
            }
        } catch {
            // ignore
        }
    }
}

interface TokenStoreOptions {
    maxCachedShards?: number;
    shardSize?: number;
    noOPFS?: boolean;
}

export async function createTokenStore(
    name: string,
    tokeniserId: string,
    datasetId: string,
    options?: TokenStoreOptions
): Promise<TokenStore> {
    const existingStore = existingStores.get(name);
    if (existingStore && existingStore.tokeniserId === tokeniserId && existingStore.datasetId === datasetId) {
        return existingStore;
    }
    if (existingStore) {
        await existingStore.dispose();
        existingStores.delete(name);
    }
    const store = new TokenStore(tokeniserId, datasetId, name, options?.maxCachedShards, options?.shardSize);
    if (!options?.noOPFS) {
        await store.init();
    }
    existingStores.set(name, store);
    return store;
}
