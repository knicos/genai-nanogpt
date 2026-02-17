import papa from 'papaparse';
import { loadParquet } from './parquet';
import { loadPDF } from './pdf';
import { loadDOCX } from './docx';

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
        default:
            return 'unknown';
    }
}

export default async function loadTextData(file: File, options?: DataOptions): Promise<string[]> {
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
            return json.map((item) =>
                typeof item === 'string' ? item : 'text' in item ? item.text : JSON.stringify(item)
            );
        } else {
            throw new Error('Expected JSON array');
        }
    }
    if (type === 'application/jsonl') {
        const data = await file.text();
        return data
            .split('\n')
            .filter((line) => line.trim() !== '')
            .map((line) => {
                try {
                    const obj = JSON.parse(line);
                    return typeof obj === 'string' ? obj : 'text' in obj ? obj.text : JSON.stringify(obj);
                } catch {
                    return line;
                }
            });
    }
    if (type === 'text/csv') {
        const data = await file.text();
        return new Promise<string[]>((resolve, reject) => {
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
                        resolve(filtered.map((row) => row[column]));
                    }
                },
                error: (error: unknown) => {
                    reject(error);
                },
            });
        });
    } else if (type === 'text/plain') {
        return [await file.text()];
    }
    throw new Error(`Unsupported file type: ${type}`);
}
