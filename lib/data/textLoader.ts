import papa from 'papaparse';
import { loadParquet } from './parquet';
import { loadPDF } from './pdf';
import { loadDOCX } from './docx';
import zip from 'jszip';
import { Conversation } from '../tokeniser/type';

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

function isConversation(obj: unknown): obj is Conversation[] {
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
}

export default async function loadTextData(
    file: File,
    options?: DataOptions,
    cb?: (progress: number) => void
): Promise<Conversation[][]> {
    const type = file.type !== '' ? file.type : getFileType(file.name);
    if (type === 'application/parquet') {
        return loadParquet(file, options?.maxSize, options?.column);
    }
    if (type === 'application/pdf') {
        return loadPDF(file, options?.maxSize);
    }
    if (type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        return loadDOCX(file);
    }
    if (type === 'application/json') {
        const data = await file.text();
        const json = JSON.parse(data);
        if (Array.isArray(json)) {
            return json.map((item) => [
                typeof item === 'string'
                    ? { role: 'text', content: item }
                    : 'text' in item
                      ? { role: 'text', content: item.text }
                      : { role: 'text', content: JSON.stringify(item) },
            ]);
        } else {
            throw new Error('Expected JSON array');
        }
    }
    if (type === 'application/jsonl') {
        const data = await file.text();

        if (cb) {
            cb(0.1);
        }

        return data
            .split('\n')
            .filter((line) => line.trim() !== '')
            .map((line, index, array) => {
                if (cb && index % 1000 === 0) {
                    cb(0.1 + (index / array.length) * 0.9);
                }
                try {
                    const obj = JSON.parse(line);
                    if (isConversation(obj)) {
                        return obj;
                    }
                    return [
                        typeof obj === 'string'
                            ? { role: 'text', content: obj }
                            : 'text' in obj
                              ? { role: 'text', content: obj.text }
                              : { role: 'text', content: JSON.stringify(obj) },
                    ];
                } catch {
                    return [{ role: 'text', content: line }];
                }
            });
    }
    if (type === 'application/zip') {
        const zipFile = await zip.loadAsync(file);
        // Loop files and call loadTextData on each, then concatenate results
        let results: Conversation[][] = [];
        const files = Object.keys(zipFile.files);

        for (let i = 0; i < files.length; i++) {
            const fileName = files[i];
            const zipEntry = zipFile.file(fileName);
            if (zipEntry) {
                const blob = await zipEntry.async('blob', (meta) => {
                    if (cb) {
                        const progress = (meta.percent / 100) * 0.9;
                        const fileProgress = progress / files.length;
                        const overallProgress = 0.1 + (fileProgress + (i / files.length) * 0.9);
                        cb(overallProgress);
                    }
                });
                const nestedResults = await loadTextData(new File([blob], fileName), options);
                if (cb) {
                    cb(0.1 + ((i + 1) / files.length) * 0.9);
                }
                results = results.concat(nestedResults);
            }
        }
        return results;
    }
    if (type === 'text/csv') {
        const data = await file.text();
        if (cb) {
            cb(0.1);
        }
        return new Promise<Conversation[][]>((resolve, reject) => {
            papa.parse<string[]>(data, {
                header: false,
                skipEmptyLines: true,
                delimiter: ',',
                complete: (results) => {
                    if (results.errors.length > 0) {
                        console.error(results.errors);
                        reject(new Error('Error parsing file'));
                    } else {
                        const column = checkForTextColumn(results.data[0], options?.column || 'text');
                        const hasHeader = options?.hasHeader ?? checkFirstRowIsHeader(results.data[0]);
                        const filtered = hasHeader ? results.data.slice(1) : results.data;
                        resolve(filtered.map((row) => [{ role: 'text', content: row[column] }]));
                    }
                },
                error: (error: unknown) => {
                    reject(error);
                },
            });
        });
    } else if (type === 'text/plain') {
        return [[{ role: 'text', content: await file.text() }]];
    }
    throw new Error(`Unsupported file type: ${type}`);
}
