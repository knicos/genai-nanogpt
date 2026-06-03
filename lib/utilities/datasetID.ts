import type { DatasetMetadata } from '../loader/types';

export default function generateDatasetID(datasets: DatasetMetadata[]): string {
    const ids = datasets.map((d) => String(d.id)).sort();

    let h1 = 0x811c9dc5;
    let h2 = 0x9e3779b9;

    const mixByte = (b: number) => {
        h1 ^= b & 0xff;
        h1 = Math.imul(h1, 0x01000193);

        h2 ^= b & 0xff;
        h2 = Math.imul(h2, 0x85ebca6b);
    };

    const mixString = (s: string) => {
        const len = s.length >>> 0;
        mixByte(len & 0xff);
        mixByte((len >>> 8) & 0xff);
        mixByte((len >>> 16) & 0xff);
        mixByte((len >>> 24) & 0xff);

        for (let i = 0; i < s.length; i++) {
            const c = s.charCodeAt(i);
            mixByte(c & 0xff);
            mixByte((c >>> 8) & 0xff);
        }
    };

    const count = ids.length >>> 0;
    mixByte(count & 0xff);
    mixByte((count >>> 8) & 0xff);
    mixByte((count >>> 16) & 0xff);
    mixByte((count >>> 24) & 0xff);

    for (const id of ids) {
        mixString(id);
    }

    return 'dataset_' + '_' + (h1 >>> 0).toString(36) + '_' + (h2 >>> 0).toString(36);
}
