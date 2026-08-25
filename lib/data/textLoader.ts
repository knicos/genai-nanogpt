import papa from 'papaparse';
import { loadPDF } from './pdf';
import { loadDOCX } from './docx';

import {
    ConversationStream,
    JSONLConversationStream,
    MemoryConversationStream,
    ZipJSONLConversationStream,
} from './stream';

export interface DataOptions {
    maxSize?: number;
    column?: string;
    hasHeader?: boolean;
}

function checkForTextColumn(header: string[], name: string): number {
    const ix = header.findIndex((col) => col.toLowerCase() === name.toLowerCase());
    return ix === -1 ? 0 : ix;
}

function checkFirstRowIsHeader(row: string[]): boolean {
    return row.every((cell) => cell.length < 64);
}

function extname(file: string): string {
    return file.split('.').pop() || '';
}

function getFileType(file: string): string {
    const ext = extname(file);
    switch (ext) {
        case 'json':
            return 'application/json';
        case 'jsonl':
            return 'application/jsonl';
        case 'parquet':
            return 'application/parquet';
        case 'csv':
            return 'text/csv';
        case 'txt':
            return 'text/plain';
        case 'pdf':
            return 'application/pdf';
        case 'docx':
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
        case 'zip':
            return 'application/zip';
        default:
            return 'unknown';
    }
}

/*function isConversation(obj: unknown): obj is Conversation[] {
    if (!Array.isArray(obj)) return false;
    const first = obj[0];
    return (
        typeof first === 'object' &&
        first !== null &&
        'role' in first &&
        'content' in first &&
        typeof first.role === 'string' &&
        typeof first.content === 'string'
    );
}*/

export default async function loadTextData(file: Blob | File, options?: DataOptions): Promise<ConversationStream> {
    const type = file.type !== '' ? file.type : file instanceof File ? getFileType(file.name) : 'application/zip';
    if (type === 'application/parquet') {
        throw new Error('unsupported_file_type');
    }
    if (type === 'application/pdf') {
        return new MemoryConversationStream(await loadPDF(file, options?.maxSize));
    }
    if (type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        return new MemoryConversationStream(await loadDOCX(file));
    }
    if (type === 'application/json') {
        const data = await file.text();
        const json = JSON.parse(data);
        if (Array.isArray(json)) {
            return new MemoryConversationStream(
                json.map((item) => [
                    typeof item === 'string'
                        ? { role: 'text', content: item }
                        : 'text' in item
                          ? { role: 'text', content: item.text }
                          : { role: 'text', content: JSON.stringify(item) },
                ])
            );
        } else {
            throw new Error('bad_format');
        }
    }
    if (type === 'application/jsonl') {
        return new JSONLConversationStream(file);
    }
    if (type === 'application/zip') {
        return new ZipJSONLConversationStream(file);
    }
    if (type === 'text/csv') {
        const data = await file.text();
        return new Promise<ConversationStream>((resolve, reject) => {
            papa.parse<string[]>(data, {
                header: false,
                skipEmptyLines: true,
                delimiter: ',',
                complete: (results) => {
                    if (results.errors.length > 0) {
                        console.error(results.errors);
                        reject(new Error('bad_format'));
                    } else {
                        const column = checkForTextColumn(results.data[0], options?.column || 'text');
                        const hasHeader = options?.hasHeader ?? checkFirstRowIsHeader(results.data[0]);
                        const filtered = hasHeader ? results.data.slice(1) : results.data;
                        resolve(
                            new MemoryConversationStream(
                                filtered.map((row) => [{ role: 'text', content: row[column] }])
                            )
                        );
                    }
                },
                error: (error: unknown) => {
                    reject(error);
                },
            });
        });
    } else if (type === 'text/plain') {
        return new MemoryConversationStream([[{ role: 'text', content: await file.text() }]]);
    }
    throw new Error('unsupported_file_type');
}
