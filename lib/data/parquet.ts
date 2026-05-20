import type { Conversation } from '../tokeniser/type';

const MAX_SIZE = 100 * 1024 * 1024; // 60 MB

export async function loadParquet(file: File, maxSize = MAX_SIZE, column = 'text'): Promise<Conversation[][]> {
    const pq = await import('@dsnp/parquetjs');
    const reader = await pq.ParquetReader.openBuffer(Buffer.from(await file.arrayBuffer()));
    const result: Conversation[][] = [];
    const cursor = reader.getCursor([[column]]);

    let totalSize = 0;

    while (true) {
        const record = (await cursor.next()) as Record<string, string>;
        if (!record || record[column] === undefined || typeof record[column] !== 'string') {
            break;
        }
        if (record[column].length === 0) {
            continue;
        }
        result.push([{ role: 'text', content: record[column] }]);
        totalSize += record[column].length;
        if (totalSize > maxSize) {
            break;
        }
    }
    reader.close();
    return result;
}
