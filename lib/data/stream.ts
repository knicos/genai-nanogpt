import type { Conversation } from '@base/tokeniser/type';
import { yieldIfNeeded } from '@base/utilities/yielder';
import { ZipReaderStream } from '@zip.js/zip.js';

export interface ConversationStream {
    begin(cb: (conv: Conversation[]) => void, yieldCb?: () => void): Promise<void>;
    step(cb: (conv: Conversation[]) => void): Promise<() => Promise<boolean>>;
}

export class MemoryConversationStream implements ConversationStream {
    private conversations: Conversation[][];

    constructor(conversations: Conversation[][]) {
        this.conversations = conversations;
    }

    async step(cb: (conv: Conversation[]) => void) {
        let i = 0;
        const next = async () => {
            if (i < this.conversations.length) {
                cb(this.conversations[i++]);
            }
            return i < this.conversations.length;
        };
        return next;
    }

    async begin(cb: (conv: Conversation[]) => void, yieldCb?: () => void) {
        let lastYield = performance.now();
        for (const conversation of this.conversations) {
            cb(conversation);
            if (yieldCb) {
                lastYield = await yieldIfNeeded(lastYield, yieldCb);
            }
        }
    }
}

function isConversationArray(value: unknown): value is Conversation[] {
    if (!Array.isArray(value)) return false;
    return value.every(
        (item) =>
            typeof item === 'object' &&
            item !== null &&
            'role' in item &&
            'content' in item &&
            typeof (item as { role: unknown }).role === 'string' &&
            typeof (item as { content: unknown }).content === 'string'
    );
}

function parseJsonlLine(line: string): Conversation[] {
    try {
        const obj = JSON.parse(line);

        if (isConversationArray(obj)) {
            return obj;
        }
        if (typeof obj === 'string') {
            return [{ role: 'text', content: obj }];
        }
        if (
            typeof obj === 'object' &&
            obj !== null &&
            'text' in obj &&
            typeof (obj as { text: unknown }).text === 'string'
        ) {
            return [{ role: 'text', content: (obj as { text: string }).text }];
        }
        return [{ role: 'text', content: JSON.stringify(obj) }];
    } catch {
        return [{ role: 'text', content: line }];
    }
}

class JSONLFromReadableStream implements ConversationStream {
    private sourceFactory: () => Promise<ReadableStream<Uint8Array>>;

    constructor(sourceFactory: () => Promise<ReadableStream<Uint8Array>>) {
        this.sourceFactory = sourceFactory;
    }

    async step(cb: (conv: Conversation[]) => void) {
        const source = await this.sourceFactory();
        const reader = source.getReader();
        const decoder = new TextDecoder();
        let remainder = '';

        const handleOne = async () => {
            const result = await reader.read();

            if (result.value || remainder.length > 0) {
                if (result.value) {
                    remainder += decoder.decode(result.value, { stream: true });
                }
                const lines = remainder.split('\n');

                if (!result.done) {
                    remainder = lines.pop() ?? '';
                }

                for (const raw of lines) {
                    const line = raw.trim();
                    if (line.length === 0) continue;
                    const conv = parseJsonlLine(line);
                    cb(conv);
                }
            }
            return !result.done;
        };

        return handleOne;
    }

    async begin(cb: (conv: Conversation[]) => void, yieldCb?: () => void) {
        const source = await this.sourceFactory();
        const reader = source.getReader();
        const decoder = new TextDecoder();
        let remainder = '';
        let lastYield = performance.now();

        return new Promise<void>((resolve) => {
            const handleOne = async () => {
                const result = await reader.read();
                if (yieldCb) {
                    lastYield = await yieldIfNeeded(lastYield, yieldCb);
                }
                if (!result.done) {
                    handleOne();
                }
                if (result.value || remainder.length > 0) {
                    if (result.value) {
                        remainder += decoder.decode(result.value, { stream: true });
                    }
                    const lines = remainder.split('\n');

                    if (!result.done) {
                        remainder = lines.pop() ?? '';
                    }

                    for (const raw of lines) {
                        const line = raw.trim();
                        if (line.length === 0) continue;
                        const conv = parseJsonlLine(line);
                        cb(conv);
                    }
                }
                if (result.done) {
                    resolve();
                }
            };

            handleOne();
        });
    }
}

export class JSONLConversationStream extends JSONLFromReadableStream {
    constructor(file: File) {
        super(async () => file.stream());
    }
}

interface ZipEntryLike {
    filename: string;
    directory?: boolean;
    readable?: ReadableStream<Uint8Array>;
}

export class ZipJSONLConversationStream extends JSONLFromReadableStream {
    constructor(file: File, preferredEntryName?: string) {
        super(async () => {
            const zipReaderStream = new ZipReaderStream<Uint8Array>();
            const entriesReader = file.stream().pipeThrough(zipReaderStream).getReader();

            let selectedReadable: ReadableStream<Uint8Array> | null = null;

            while (true) {
                const { value, done } = await entriesReader.read();
                if (done) break;

                const entry = value as ZipEntryLike;
                if (entry.directory) continue;
                if (!entry.readable) continue;

                if (preferredEntryName && entry.filename === preferredEntryName) {
                    selectedReadable = entry.readable;
                    break;
                }

                if (!preferredEntryName && entry.filename.toLowerCase().endsWith('.jsonl')) {
                    selectedReadable = entry.readable;
                    break;
                }
            }

            await entriesReader.cancel();

            if (!selectedReadable) {
                throw new Error('No JSONL entry found in ZIP');
            }

            return selectedReadable;
        });
    }
}
