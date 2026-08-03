import type { Conversation } from '@base/tokeniser/type';
import { ZipReaderStream } from '@zip.js/zip.js';

export interface ConversationCursor {
    next(): Promise<Conversation[] | null>;
}

export interface ConversationStream {
    cursor(): ConversationCursor;
}

export class MemoryConversationStream implements ConversationStream {
    private conversations: Conversation[][];

    constructor(conversations: Conversation[][]) {
        this.conversations = conversations;
    }

    cursor(): ConversationCursor {
        let index = 0;
        const conversations = this.conversations;
        return {
            async next(): Promise<Conversation[] | null> {
                if (index < conversations.length) {
                    return conversations[index++];
                }
                return null;
            },
        };
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

    cursor(): ConversationCursor {
        let initialized = false;
        let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
        const decoder = new TextDecoder();
        let remainder = '';
        let done = false;
        const queue: Conversation[][] = [];

        const init = async () => {
            if (!initialized) {
                const source = await this.sourceFactory();
                reader = source.getReader();
                initialized = true;
            }
        };

        const fillQueue = async () => {
            await init();

            while (queue.length === 0 && !done) {
                const result = await reader!.read();

                if (result.done) {
                    remainder += decoder.decode();
                    done = true;

                    const finalLine = remainder.trim();
                    if (finalLine.length > 0) {
                        queue.push(parseJsonlLine(finalLine));
                    }
                    remainder = '';
                    break;
                }

                remainder += decoder.decode(result.value, { stream: true });

                const lines = remainder.split('\n');
                remainder = lines.pop() ?? '';

                for (const raw of lines) {
                    const line = raw.trim();
                    if (line.length === 0) continue;
                    queue.push(parseJsonlLine(line));
                }
            }
        };

        return {
            async next(): Promise<Conversation[] | null> {
                if (queue.length > 0) return queue.shift() ?? null;
                await fillQueue();
                return queue.shift() ?? null;
            },
        };
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
