import { File as NodeFile } from 'node:buffer';
import { describe, it } from 'vitest';
import { ZipWriter, BlobWriter, TextReader } from '@zip.js/zip.js';
import loadTextData from './textLoader';

async function collectConversations(file: Blob) {
    const result = await loadTextData(file);
    const conversations: { role: string; content: string }[][] = [];
    await result.begin((conv) => {
        conversations.push(conv);
    });
    return conversations;
}

describe('Text loading', () => {
    it('should load a json file', async ({ expect }) => {
        const file = new NodeFile([JSON.stringify([{ text: 'Hello' }, { text: 'World' }])], 'test.json', {
            type: 'application/json',
        });
        const conversations = await collectConversations(file as unknown as File);
        expect(conversations).toEqual([[{ role: 'text', content: 'Hello' }], [{ role: 'text', content: 'World' }]]);
    });

    it('should load a jsonl file', async ({ expect }) => {
        const file = new NodeFile(
            [JSON.stringify({ text: 'Hello' }) + '\n' + JSON.stringify({ text: 'World' })],
            'test.jsonl',
            {
                type: 'application/jsonl',
            }
        );
        const conversations = await collectConversations(file as unknown as File);
        expect(conversations).toEqual([[{ role: 'text', content: 'Hello' }], [{ role: 'text', content: 'World' }]]);
    });

    it('should load a csv file', async ({ expect }) => {
        const file = new NodeFile(['text,title,other\nHello,some,thing\nWorld,another,thing'], 'test.csv', {
            type: 'text/csv',
        });
        const conversations = await collectConversations(file as unknown as File);
        expect(conversations).toEqual([[{ role: 'text', content: 'Hello' }], [{ role: 'text', content: 'World' }]]);
    });

    it('should load a jsonl conversation file', async ({ expect }) => {
        const file = new NodeFile(
            [
                JSON.stringify([
                    { role: 'user', content: 'Hello' },
                    { role: 'assistant', content: 'Hi there!' },
                ]) +
                    '\n' +
                    JSON.stringify([
                        { role: 'user', content: 'World' },
                        { role: 'assistant', content: 'Hello!' },
                    ]),
            ],
            'test.jsonl',
            {
                type: 'application/jsonl',
            }
        );

        const conversations = await collectConversations(file as unknown as File);
        expect(conversations).toEqual([
            [
                { role: 'user', content: 'Hello' },
                { role: 'assistant', content: 'Hi there!' },
            ],
            [
                { role: 'user', content: 'World' },
                { role: 'assistant', content: 'Hello!' },
            ],
        ]);
    });

    // Seems to be a bug inside zip-js that causes this test to fail in Node.js, but it works in the browser. Skipping for now.
    it.skip('should load a zipped jsonl file', async ({ expect }) => {
        const blobWriter = new BlobWriter('application/zip');
        const writer = new ZipWriter(blobWriter);
        await writer.add(
            'dataset.jsonl',
            new TextReader(JSON.stringify({ text: 'Hello' }) + '\n' + JSON.stringify({ text: 'World' }))
        );
        const zippedBlob = await writer.close();

        const conversations = await collectConversations(zippedBlob);
        expect(conversations).toEqual([[{ role: 'text', content: 'Hello' }], [{ role: 'text', content: 'World' }]]);
    });
});
