const MAX_SIZE = 100 * 1024 * 1024; // 60 MB

export async function loadPDF(file: File, maxSize = MAX_SIZE): Promise<string[]> {
    const pdfjsLib = await import('pdfjs-dist/legacy/build/pdf.mjs');
    const loadingTask = pdfjsLib.getDocument({ data: await file.arrayBuffer() });
    const pdf = await loadingTask.promise;
    const numPages = pdf.numPages;

    const result: string[] = [];
    let totalSize = 0;
    for (let i = 1; i <= numPages; i++) {
        const page = await pdf.getPage(i);
        const textContent = await page.getTextContent();
        const textItems = textContent.items as Array<{ str: string }>;
        const pageText = textItems.map((item) => item.str).join(' ');
        result.push(pageText);
        totalSize += pageText.length;
        if (totalSize > maxSize) break;
    }

    return result;
}
