import { describe, it } from 'vitest';
import { loadPDF } from './pdf';
import fs from 'fs';
import path from 'path';

describe('PDF loading', () => {
    it('should load a PDF file', async ({ expect }) => {
        const filePath = path.resolve(__dirname, './test.pdf');
        const file = fs.readFileSync(filePath);
        const result = await loadPDF(new Uint8Array(file));
        expect(result).toBeInstanceOf(Array);
    });
});
