import papa from 'papaparse';
import { loadParquet } from './parquet';

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

export default async function loadTextData(file: File, options?: DataOptions): Promise<string[]> {
    const type = file.type;
    if (type === 'application/parquet') {
        return loadParquet(file, options?.maxSize, options?.column);
    }
    if (type === 'text/csv') {
        const data = !('FileReaderSync' in global) ? await file.text() : file;
        return new Promise<string[]>((resolve, reject) => {
            papa.parse<string[]>(data, {
                header: false,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        reject(new Error('Error parsing file'));
                    } else {
                        const column = checkForTextColumn(results.data[0], options?.column || 'text');
                        const hasHeader = options?.hasHeader ?? checkFirstRowIsHeader(results.data[0]);
                        const filtered = hasHeader ? results.data.slice(1) : results.data;
                        resolve(filtered.map((row) => row[column]));
                    }
                },
                error: (error) => {
                    reject(error);
                },
            });
        });
    } else if (type === 'text/plain') {
        return [await file.text()];
    }
    throw new Error(`Unsupported file type: ${type}`);
}
