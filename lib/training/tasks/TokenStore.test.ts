import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { TokenStore, createTokenStore, deleteTokenStore, getTokenStore } from './TokenStore';

// Helpers for tests
function makeRandomShard(len: number, seed = 1): Uint16Array {
    const a = new Uint16Array(len);
    for (let i = 0; i < len; i++) a[i] = (seed + i) & 0xffff;
    return a;
}

async function waitForCondition(fn: () => boolean | Promise<boolean>, timeout = 5000, interval = 50) {
    const start = Date.now();

    while (true) {
        try {
            if (await fn()) return;
        } catch {
            // swallow transient errors from the predicate and retry until timeout
        }
        if (Date.now() - start > timeout) throw new Error('timeout waiting for condition');

        await new Promise((r) => setTimeout(r, interval));
    }
}

/**
 * Minimal in-memory OPFS-like mock implementing the parts TokenStore uses:
 * - navigator.storage.getDirectory()
 * - DirectoryHandle.getFileHandle(name, { create })
 * - FileHandle.createWritable(), FileHandle.getFile()
 * - File-like object: arrayBuffer(), text()
 * - DirectoryHandle.values() iterator
 * - DirectoryHandle.removeEntry(name)
 */
function makeOpfsMock() {
    interface FileRecord {
        data?: Uint8Array;
        text?: string;
    }

    interface FileCounters {
        writes: number;
        reads: number;
    }

    const directories = new Map<string, Map<string, FileRecord>>();
    const counters = new Map<string, FileCounters>();

    const keyFor = (dirName: string, fileName: string) => `${dirName}/${fileName}`;
    const bumpWrite = (dirName: string, fileName: string) => {
        const k = keyFor(dirName, fileName);
        const c = counters.get(k) ?? { writes: 0, reads: 0 };
        c.writes += 1;
        counters.set(k, c);
    };
    const bumpRead = (dirName: string, fileName: string) => {
        const k = keyFor(dirName, fileName);
        const c = counters.get(k) ?? { writes: 0, reads: 0 };
        c.reads += 1;
        counters.set(k, c);
    };

    const getReadCount = (dirName: string, fileName: string) => counters.get(keyFor(dirName, fileName))?.reads ?? 0;

    const root = {
        async getDirectoryHandle(name: string, opts?: { create?: boolean }) {
            if (!directories.has(name)) {
                if (!opts?.create) throw new Error('NotFound');
                directories.set(name, new Map());
            }
            const dirMap = directories.get(name)!;
            const dirHandle = {
                async getFileHandle(fileName: string, opts?: { create?: boolean }) {
                    if (!dirMap.has(fileName)) {
                        if (!opts?.create) throw new Error('NotFound');
                        dirMap.set(fileName, {});
                    }
                    const fileRecord = dirMap.get(fileName)!;
                    const fileHandle = {
                        async createWritable(opts?: { keepExistingData?: boolean }) {
                            // buffer writes until close
                            const chunks: Uint8Array[] = [];
                            return {
                                async write(data: Blob | ArrayBuffer | ArrayBufferView | string) {
                                    bumpWrite(name, fileName);
                                    if (typeof data === 'string') {
                                        if (!opts?.keepExistingData) {
                                            fileRecord.text = '';
                                        }
                                        fileRecord.text = (fileRecord.text ?? '') + data;
                                    } else if ((data as Blob).arrayBuffer) {
                                        // Blob
                                        const ab = await (data as Blob).arrayBuffer();
                                        chunks.push(new Uint8Array(ab));
                                    } else if (data instanceof ArrayBuffer) {
                                        chunks.push(new Uint8Array(data));
                                    } else if (ArrayBuffer.isView(data)) {
                                        const view = data as ArrayBufferView;
                                        chunks.push(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
                                    } else {
                                        throw new Error('unsupported write type');
                                    }
                                },
                                async close() {
                                    if (chunks.length > 0) {
                                        // merge
                                        let total = 0;
                                        for (const c of chunks) total += c.length;
                                        const out = new Uint8Array(total);
                                        let off = 0;
                                        for (const c of chunks) {
                                            out.set(c, off);
                                            off += c.length;
                                        }
                                        fileRecord.data = out;
                                    }
                                },
                            };
                        },
                        async getFile() {
                            bumpRead(name, fileName);
                            return {
                                async arrayBuffer() {
                                    if (!fileRecord.data) return new ArrayBuffer(0);
                                    return fileRecord.data.buffer.slice(0);
                                },
                                async text() {
                                    return fileRecord.text ?? '';
                                },
                            };
                        },
                    };
                    return fileHandle;
                },
                async removeEntry(name: string) {
                    dirMap.delete(name);
                },
                values() {
                    const iter = (function* (m: Map<string, FileRecord>) {
                        for (const k of m.keys()) yield { name: k };
                    })(dirMap);
                    return iter;
                },
            };
            return dirHandle;
        },
        async removeEntry(name: string) {
            directories.delete(name);
        },
    };

    return { root, directories, getReadCount };
}

function toArray(a: Uint16Array | Uint8Array) {
    return Array.from(a);
}

describe('TokenStore (memory-only)', () => {
    it('tracks shard/token metadata and supports cross-shard slicing', async () => {
        const store = new TokenStore('tok-a', 'ds-a', 'memory-meta', 8, 5);
        await store.appendShard(makeRandomShard(5, 10));
        await store.appendShard(makeRandomShard(3, 100));

        expect(store.getShardCount()).toBe(2);
        expect(store.getShardLength(0)).toBe(5);
        expect(store.getShardLength(1)).toBe(3);
        expect(store.getTokenCount()).toBe(8);

        const slice = await store.slice(3, 7);
        expect(toArray(slice)).toEqual([13, 14, 100, 101]);
    });

    it('rejects invalid shard and slice ranges', async () => {
        const store = new TokenStore('tok-b', 'ds-b', 'memory-ranges', 8, 4);
        await store.appendShard(makeRandomShard(4, 1));

        expect(() => store.getShardLength(-1)).toThrow(/out of range/i);
        expect(() => store.getShardLength(2)).toThrow(/out of range/i);
        await expect(store.getShard(-1)).rejects.toThrow(/out of range/i);
        await expect(store.getShard(1)).rejects.toThrow(/out of range/i);
        await expect(store.slice(-1, 2)).rejects.toThrow(/invalid slice range/i);
        await expect(store.slice(2, 2)).rejects.toThrow(/invalid slice range/i);
        await expect(store.slice(0, 10)).rejects.toThrow(/invalid slice range/i);
    });

    it('requires all non-final shards to be full', async () => {
        const store = new TokenStore('tok-c', 'ds-c', 'memory-shape', 8, 5);
        await store.appendShard(makeRandomShard(3, 1));
        await expect(store.appendShard(makeRandomShard(2, 10))).rejects.toThrow(/previous shard was not full/i);
    });

    it('evicts old shards without OPFS and still serves latest cached shards', async () => {
        const store = new TokenStore('tok-d', 'ds-d', 'memory-evict', 1, 16);
        const s0 = makeRandomShard(16, 1);
        const s1 = makeRandomShard(16, 100);

        await store.appendShard(s0);
        await store.appendShard(s1);

        expect(store.getShardCount()).toBe(2);
        const got1 = await store.getShard(1);
        expect(got1).toBeInstanceOf(Uint16Array);
        expect(got1.length).toBe(16);
        expect(toArray(got1)).toEqual(toArray(s1));

        let got0: Uint16Array | null = null;
        let threw = false;
        try {
            got0 = await store.getShard(0);
        } catch {
            threw = true;
        }
        if (!threw) {
            expect(got0).toBeInstanceOf(Uint16Array);
            expect(toArray(got0!)).toEqual(toArray(s0));
        } else {
            expect(threw).toBe(true);
        }
    });
});

describe('TokenStore (OPFS-backed, mocked)', () => {
    let opfs: ReturnType<typeof makeOpfsMock>;
    const storeId = 'test-opfs-store';

    beforeEach(() => {
        opfs = makeOpfsMock();
        globalThis.navigator = {
            storage: {
                getDirectory: async () => opfs.root as unknown as FileSystemDirectoryHandle,
            } as unknown as StorageManager,
        } as unknown as Navigator;
    });

    afterEach(async () => {
        // cleanup
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
        await deleteTokenStore(storeId);
    });

    it('writes shards + manifest to OPFS and reloads data with matching tokeniser/dataset', async () => {
        const store = await createTokenStore(storeId, 'x', 'x', {
            maxCachedShards: 1,
            shardSize: 32,
        });
        const s0 = makeRandomShard(32, 1);
        const s1 = makeRandomShard(20, 1000);
        await store.appendShard(s0);
        await store.appendShard(s1);

        await waitForCondition(() => {
            const dirs = (opfs.directories.get(storeId) ?? new Map()).keys();
            const k = Array.from(dirs);
            return k.includes('shard-0.bin') && k.includes('shard-1.bin') && k.includes('manifest.json');
        }, 2000);

        const dir = opfs.directories.get(storeId)!;
        expect(dir.has('shard-0.bin')).toBeTruthy();
        expect(dir.has('shard-1.bin')).toBeTruthy();
        expect(dir.has('manifest.json')).toBeTruthy();

        const store2 = new TokenStore('x', 'x', storeId, 1);
        await store2.init();
        expect(store2.getShardCount()).toBe(2);
        expect(store2.getTokenCount()).toBe(52);

        const r0 = await store2.getShard(0);
        const r1 = await store2.getShard(1);
        expect(toArray(r0)).toEqual(toArray(s0));
        expect(toArray(r1)).toEqual(toArray(s1));
    });

    it('clears incompatible persisted store when tokeniser or dataset identity mismatches', async () => {
        const original = await createTokenStore(storeId, 'tokeniser-A', 'dataset-A', {
            maxCachedShards: 1,
            shardSize: 8,
        });
        await original.appendShard(makeRandomShard(8, 1));

        await waitForCondition(() => {
            const d = opfs.directories.get(storeId) ?? new Map();
            return d.has('shard-0.bin') && d.has('manifest.json');
        }, 2000);

        const reloadedWithMismatch = new TokenStore('tokeniser-B', 'dataset-A', storeId, 1, 8);
        await reloadedWithMismatch.init();

        const dir = opfs.directories.get(storeId) ?? new Map();
        expect(Array.from(dir.keys())).toEqual([]);
        expect(reloadedWithMismatch.getShardCount()).toBe(0);
    });

    it('supports random shard access by lazily loading evicted shards from OPFS', async () => {
        const store = await createTokenStore(storeId, 'tok-random', 'ds-random', {
            maxCachedShards: 1,
            shardSize: 8,
        });

        const s0 = makeRandomShard(8, 11);
        const s1 = makeRandomShard(8, 111);
        const s2 = makeRandomShard(8, 211);
        await store.appendShard(s0);
        await store.appendShard(s1);
        await store.appendShard(s2);

        await waitForCondition(() => {
            const d = opfs.directories.get(storeId) ?? new Map();
            return d.has('shard-0.bin') && d.has('shard-1.bin') && d.has('shard-2.bin');
        }, 2000);

        const order = [2, 0, 1, 2, 0];
        const expected = [s2, s0, s1, s2, s0];
        for (let i = 0; i < order.length; i++) {
            const got = await store.getShard(order[i]);
            expect(toArray(got)).toEqual(toArray(expected[i]));
        }

        expect(opfs.getReadCount(storeId, 'shard-0.bin')).toBeGreaterThan(0);
        expect(opfs.getReadCount(storeId, 'shard-1.bin')).toBeGreaterThan(0);
    });

    it('persists and reloads masks alongside shards', async () => {
        const store = await createTokenStore(storeId, 'tok-mask', 'ds-mask', {
            maxCachedShards: 1,
            shardSize: 6,
        });
        const shard = makeRandomShard(6, 500);
        const mask = new Uint8Array([1, 0, 1, 1, 0, 1]);
        await store.appendShard(shard, mask);

        const store2 = new TokenStore('tok-mask', 'ds-mask', storeId, 1, 6);
        await store2.init();

        expect(store2.hasMask()).toBe(true);
        const loadedMask = await store2.getMask(0);
        expect(loadedMask).toBeDefined();
        expect(toArray(loadedMask!)).toEqual(toArray(mask));
    });

    it('clear resets in-memory state and persists empty manifest state', async () => {
        const store = await createTokenStore(storeId, 'tok-clear', 'ds-clear', {
            maxCachedShards: 1,
            shardSize: 8,
        });
        await store.appendShard(makeRandomShard(8, 10));
        await store.clear();

        expect(store.getShardCount()).toBe(0);
        expect(store.getTokenCount()).toBe(0);

        const reloaded = new TokenStore('tok-clear', 'ds-clear', storeId, 1, 8);
        await reloaded.init();
        expect(reloaded.getShardCount()).toBe(0);
    });

    it('dispose removes OPFS files and store directory', async () => {
        const store = await createTokenStore(storeId, 'store-to-dispose', 'dataset-to-dispose', {
            maxCachedShards: 1,
            shardSize: 8,
        });
        await store.appendShard(makeRandomShard(8, 50));

        await waitForCondition(() => {
            const d = opfs.directories.get(storeId) ?? new Map();
            return d.has('shard-0.bin') && d.has('manifest.json');
        }, 2000);

        await store.dispose();

        await new Promise((r) => setTimeout(r, 10));
        expect(opfs.directories.has(storeId)).toBe(false);
    });
});

describe('TokenStore registry helpers', () => {
    let opfs: ReturnType<typeof makeOpfsMock>;
    const storeId = 'registry-store';

    beforeEach(() => {
        opfs = makeOpfsMock();
        globalThis.navigator = {
            storage: {
                getDirectory: async () => opfs.root as unknown as FileSystemDirectoryHandle,
            } as unknown as StorageManager,
        } as unknown as Navigator;
    });

    afterEach(async () => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (globalThis as any).navigator;
        await deleteTokenStore(storeId);
    });

    it('reuses an existing store only when name + tokeniserId + datasetId all match', async () => {
        const a = await createTokenStore(storeId, 'tok-1', 'ds-1', { noOPFS: true });
        const b = await createTokenStore(storeId, 'tok-1', 'ds-1', { noOPFS: true });
        expect(a).toBe(b);

        const c = await createTokenStore(storeId, 'tok-2', 'ds-1', { noOPFS: true });
        expect(c).not.toBe(a);
    });

    it('getTokenStore returns null for identity mismatch and instance for exact match', async () => {
        const store = await createTokenStore(storeId, 'tok-get', 'ds-get', { noOPFS: true });

        expect(getTokenStore(storeId, 'tok-get', 'ds-get')).toBe(store);
        expect(getTokenStore(storeId, 'tok-other', 'ds-get')).toBeNull();
        expect(getTokenStore(storeId, 'tok-get', 'ds-other')).toBeNull();
    });

    it('deleteTokenStore removes persisted data even if the store is not in registry', async () => {
        const store = await createTokenStore(storeId, 'tok-delete', 'ds-delete', { shardSize: 4 });
        await store.appendShard(makeRandomShard(4, 1));
        await deleteTokenStore(storeId);

        expect(opfs.directories.has(storeId)).toBe(false);
    });
});
