import { File as NodeFile } from 'node:buffer';
import { describe, it } from 'vitest';
import loadTextData from './textLoader';

describe('Text loading', () => {
    it('should load a json file', async ({ expect }) => {
        const file = new NodeFile([JSON.stringify([{ text: 'Hello' }, { text: 'World' }])], 'test.json', {
            type: 'application/json',
        });
        const result = await loadTextData(file as unknown as File);
        const cursor = result.cursor();
        const first = await cursor.next();
        const second = await cursor.next();
        expect(first).toEqual([{ role: 'text', content: 'Hello' }]);
        expect(second).toEqual([{ role: 'text', content: 'World' }]);
    });

    it('should load a jsonl file', async ({ expect }) => {
        const file = new NodeFile(
            [JSON.stringify({ text: 'Hello' }) + '\n' + JSON.stringify({ text: 'World' })],
            'test.jsonl',
            {
                type: 'application/jsonl',
            }
        );
        const result = await loadTextData(file as unknown as File);
        const stream = result.cursor();
        const first = await stream.next();
        const second = await stream.next();
        expect(first).toEqual([{ role: 'text', content: 'Hello' }]);
        expect(second).toEqual([{ role: 'text', content: 'World' }]);
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
        const result = await loadTextData(file as unknown as File);
        const stream = result.cursor();
        const first = await stream.next();
        const second = await stream.next();
        expect(first).toEqual([
            { role: 'user', content: 'Hello' },
            { role: 'assistant', content: 'Hi there!' },
        ]);
        expect(second).toEqual([
            { role: 'user', content: 'World' },
            { role: 'assistant', content: 'Hello!' },
        ]);
    });
});
