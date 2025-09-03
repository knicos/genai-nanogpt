import jszip from 'jszip';

export async function loadDOCX(file: Blob | Uint8Array): Promise<string[]> {
    const zip = await jszip.loadAsync(file);

    const doc = await zip.file('word/document.xml')?.async('string');

    if (!doc) throw new Error('Failed to load document.xml');

    const text = extractTextFromDOCX(doc);
    return text.split('\n').filter((line) => line.trim().length > 10);
}

function extractTextFromDOCX(xml: string): string {
    const parser = new DOMParser();
    const doc = parser.parseFromString(xml, 'application/xml');
    const texts = Array.from(doc.getElementsByTagName('w:t')).map((node) => node.textContent);
    return texts.join('\n');
}
