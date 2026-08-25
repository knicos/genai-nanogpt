import type { Conversation } from '../tokeniser/type';

const MAX_SIZE = 100 * 1024 * 1024; // 60 MB
const PDFJS_BASE_URL = 'https://store.gen-ai.fi/llm/deps';

export async function loadPDF(file: Blob | Uint8Array, maxSize = MAX_SIZE): Promise<Conversation[][]> {
    const pdfjsLib = await import('pdfjs-dist/legacy/build/pdf.mjs');

    if (!pdfjsLib.GlobalWorkerOptions.workerSrc) {
        pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(`${PDFJS_BASE_URL}/pdf.worker.min.mjs`).toString();
    }

    const loadingTask = pdfjsLib.getDocument({ data: file instanceof Blob ? await file.arrayBuffer() : file });
    const pdf = await loadingTask.promise;
    const numPages = pdf.numPages;

    const result: Conversation[][] = [];
    let totalSize = 0;
    for (let i = 1; i <= numPages; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        const textItems = textContent.items as { str: string }[];
        const filtered = textItems.filter((item) => item.str.trim().length > 10);
        const pageText = filtered.map((item) => item.str).join(' ');
        result.push([{ role: 'text', content: pageText }]);
        totalSize += pageText.length;
        if (totalSize > maxSize) break;
    }

    return result;
}
