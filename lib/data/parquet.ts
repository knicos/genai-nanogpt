const MAX_SIZE = 100 * 1024 * 1024; // 60 MB

export async function loadParquet(file: File, maxSize = MAX_SIZE, column = 'text'): Promise<string[]> {
    const pq = await import('@dsnp/parquetjs');
    const reader = await pq.ParquetReader.openBuffer(Buffer.from(await file.arrayBuffer()));
    const result: string[] = [];
    const cursor = reader.getCursor([[column]]);

    let totalSize = 0;

    while (true) {
        const record = (await cursor.next()) as { [key: string]: string };
        if (!record || !record[column] || typeof record[column] !== 'string') break;
        result.push(record[column]);
        totalSize += record[column].length;
        if (totalSize > maxSize) break;
    }
    reader.close();
    return result;
}
