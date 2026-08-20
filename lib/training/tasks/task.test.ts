import { describe, it, vi, afterEach } from 'vitest';
import { tokensFromStreams } from './tokenStream';
import CharTokeniser from '@base/tokeniser/CharTokeniser';
import { Conversation } from '@base/tokeniser/type';
import { MemoryConversationStream } from '@base/data/stream';

async function collectAllTokens(tokens: { getShardCount(): number; getShard(index: number): Promise<Uint16Array> }) {
    const parts: number[] = [];
    for (let i = 0; i < tokens.getShardCount(); i++) {
        parts.push(...Array.from(await tokens.getShard(i)));
    }
    return new Uint16Array(parts);
}

describe('tokensFromStreams', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('tokenises multiple tasks and keeps task ordering in output sequence', async ({ expect }) => {
        const data1: Conversation[][] = [
            [
                { role: 'text', content: 'Hello world.' },
                { role: 'text', content: 'How are you?' },
            ],
        ];
        const data2: Conversation[][] = [
            [
                { role: 'text', content: 'This is a test.' },
                { role: 'text', content: 'Testing 123.' },
            ],
        ];
        const stream1 = new MemoryConversationStream(data1);
        const stream2 = new MemoryConversationStream(data2);
        const tasks = [stream1, stream2];

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train([stream1, stream2]);
        tokeniser.datasetID = 'test-dataset1';
        const { trainingTokens: tokens, validationTokens } = await tokensFromStreams(tasks, tokeniser, {
            noOPFS: true,
        });

        expect(validationTokens).toBeUndefined();
        expect(tokens.getTokenCount()).toBeGreaterThan(data1.length + data2.length);
        const decodedText = tokeniser.decodeConversation(await tokens.getShard(0));

        expect(decodedText[0].content).toContain('Hello world.How are you?');
        expect(decodedText[0].role).toBe('text');
        expect(decodedText[1].content).toContain('This is a test.Testing 123.');
        expect(decodedText[1].role).toBe('text');
    });

    it('places all tokens into validation when validationSplit is 1', async ({ expect }) => {
        const data1: Conversation[][] = [
            [
                { role: 'text', content: 'Hello world.' },
                { role: 'text', content: 'How are you?' },
            ],
        ];
        const data2: Conversation[][] = [
            [
                { role: 'text', content: 'This is a test.' },
                { role: 'text', content: 'Testing 123.' },
            ],
        ];
        const stream1 = new MemoryConversationStream(data1);
        const stream2 = new MemoryConversationStream(data2);
        const tasks = [stream1, stream2];

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train([stream1, stream2]);
        tokeniser.datasetID = 'test-dataset-val-100';

        vi.spyOn(Math, 'random').mockReturnValue(0);

        const { trainingTokens: tokens, validationTokens } = await tokensFromStreams(tasks, tokeniser, {
            validationSplit: 1,
            noOPFS: true,
        });

        expect(validationTokens).toBeDefined();
        expect(validationTokens!.getTokenCount()).toBeGreaterThan(0);

        const decodedValidation = tokeniser.decode(await collectAllTokens(validationTokens!));
        expect(decodedValidation).toContain('Hello world.How are you?');
        expect(decodedValidation).toContain('This is a test.Testing 123.');

        // Current implementation may still create an empty training shard; contract intent is no training content.
        expect(tokens.getTokenCount()).toBe(0);
    });

    it('does not create validation store when validationSplit is 0', async ({ expect }) => {
        const data: Conversation[][] = [[{ role: 'text', content: 'Only training split.' }]];
        const stream = new MemoryConversationStream(data);

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train([stream]);
        tokeniser.datasetID = 'test-dataset-val-0';

        const { trainingTokens, validationTokens } = await tokensFromStreams([stream], tokeniser, {
            validationSplit: 0,
            noOPFS: true,
        });

        expect(validationTokens).toBeUndefined();
        expect(trainingTokens.getTokenCount()).toBeGreaterThan(0);
    });

    it('routes data to both sets deterministically with validationSplit using Math.random', async ({ expect }) => {
        const data: Conversation[][] = [
            [{ role: 'text', content: 'conv-A' }],
            [{ role: 'text', content: 'conv-B' }],
            [{ role: 'text', content: 'conv-C' }],
            [{ role: 'text', content: 'conv-D' }],
        ];
        const stream = new MemoryConversationStream(data);

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train([stream]);
        tokeniser.datasetID = 'test-dataset-val-mixed';

        const randomValues = [0.1, 0.9, 0.2, 0.8];
        let idx = 0;
        vi.spyOn(Math, 'random').mockImplementation(() => randomValues[idx++] ?? 0.9);

        const { trainingTokens, validationTokens } = await tokensFromStreams([stream], tokeniser, {
            validationSplit: 0.5,
            noOPFS: true,
        });

        expect(validationTokens).toBeDefined();
        const trainingDecoded = tokeniser.decode(await collectAllTokens(trainingTokens));
        const validationDecoded = tokeniser.decode(await collectAllTokens(validationTokens!));

        // A and C go to validation; B and D go to training based on mocked random values.
        expect(validationDecoded).toContain('conv-A');
        expect(validationDecoded).toContain('conv-C');
        expect(trainingDecoded).toContain('conv-B');
        expect(trainingDecoded).toContain('conv-D');
    });

    it('generates mask aligned to tokens for assistant/user conversations', async ({ expect }) => {
        const data1: Conversation[][] = [
            [
                { role: 'user', content: 'Hello world.' },
                { role: 'assistant', content: 'How are you?' },
            ],
        ];
        const data2: Conversation[][] = [
            [
                { role: 'user', content: 'This is a test.' },
                { role: 'assistant', content: 'Testing 123.' },
            ],
        ];
        const stream1 = new MemoryConversationStream(data1);
        const stream2 = new MemoryConversationStream(data2);

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train([stream1, stream2]);
        tokeniser.datasetID = 'test-dataset2';
        const { trainingTokens: tokens } = await tokensFromStreams([stream1, stream2], tokeniser, {
            masking: true,
            noOPFS: true,
        });

        const decodedText = tokeniser.decodeConversation(await tokens.getShard(0));

        const mask = await tokens.getMask(0);
        const shard = await tokens.getShard(0);

        expect(mask?.length).toBe(shard.length);
        expect(mask?.some((m) => m === 0)).toBe(true);
        expect(mask?.some((m) => m === 1)).toBe(true);

        expect(decodedText[0].content).toContain('Hello world.');
        expect(decodedText[0].role).toBe('user');
        expect(decodedText[1].content).toContain('How are you?');
        expect(decodedText[1].role).toBe('assistant');
    });

    it('supports masking in validation store when split sends all data to validation', async ({ expect }) => {
        const data: Conversation[][] = [
            [
                { role: 'user', content: 'Question?' },
                { role: 'assistant', content: 'Answer.' },
            ],
        ];
        const stream = new MemoryConversationStream(data);

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train([stream]);
        tokeniser.datasetID = 'test-dataset-mask-val';

        vi.spyOn(Math, 'random').mockReturnValue(0);

        const { validationTokens } = await tokensFromStreams([stream], tokeniser, {
            masking: true,
            validationSplit: 1,
            noOPFS: true,
        });

        expect(validationTokens).toBeDefined();
        expect(validationTokens!.hasMask()).toBe(true);
        const vShard = await validationTokens!.getShard(0);
        const vMask = await validationTokens!.getMask(0);
        expect(vMask?.length).toBe(vShard.length);
    });

    it('handles shard rollover and preserves full decoded output across shards', async ({ expect }) => {
        const data1: Conversation[][] = [[{ role: 'text', content: 'short first sentence' }]];

        for (let i = 0; i < 20; i++) {
            data1.push([{ role: 'text', content: `This is sentence number ${i}. ` + 'A'.repeat(80) }]);
        }
        const stream1 = new MemoryConversationStream(data1);

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train([stream1]);
        tokeniser.datasetID = 'test-dataset3';

        const { trainingTokens: tokens } = await tokensFromStreams([stream1], tokeniser, {
            noOPFS: true,
            shardSize: 128,
            maxCachedShards: 10_000,
        });

        expect(tokens.getTokenCount()).toBeGreaterThan(data1.length);
        expect(tokens.getShardCount()).toBeGreaterThan(1);

        const merged = await collectAllTokens(tokens);
        const decoded = tokeniser.decode(merged);

        for (const d of data1) {
            expect(decoded).toContain(d.map((c) => c.content).join(''));
        }
    });

    it('handles multiple streams within a single task', async ({ expect }) => {
        const s1Data: Conversation[][] = [[{ role: 'text', content: 'stream-1 message' }]];
        const s2Data: Conversation[][] = [[{ role: 'text', content: 'stream-2 message' }]];
        const stream1 = new MemoryConversationStream(s1Data);
        const stream2 = new MemoryConversationStream(s2Data);

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train([stream1, stream2]);
        tokeniser.datasetID = 'test-dataset-multi-stream';

        const { trainingTokens: tokens } = await tokensFromStreams([stream1, stream2], tokeniser, { noOPFS: true });
        const decoded = tokeniser.decode(await collectAllTokens(tokens));

        expect(decoded).toContain('stream-1 message');
        expect(decoded).toContain('stream-2 message');
    });

    it('can generate tokens from multiple different tasks', async ({ expect }) => {
        const data1: Conversation[][] = [
            [
                { role: 'text', content: 'Hello world.' },
                { role: 'text', content: 'How are you?' },
            ],
        ];
        const data2: Conversation[][] = [
            [
                { role: 'text', content: 'This is a test. You now must complete the sentence.' },
                { role: 'text', content: 'Testing 123. 123 Testing.' },
            ],
        ];
        const stream1 = new MemoryConversationStream(data1);
        const stream2 = new MemoryConversationStream(data2);

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train([stream1, stream2]);
        tokeniser.datasetID = 'test-dataset4';

        const { trainingTokens: tokens } = await tokensFromStreams([stream1, stream2], tokeniser, { noOPFS: true });

        expect(tokens.getTokenCount()).toBeGreaterThan(data1.length + data2.length);
        const decodedText = await tokeniser.decode(await tokens.getShard(0));

        expect(decodedText).toContain(
            '<bos>Hello world.How are you?<eos><bos>This is a test. You now must complete the sentence.Testing 123. 123 Testing.<eos>'
        );
    });

    it('throws when one encoded conversation exceeds shard size', async ({ expect }) => {
        const data: Conversation[][] = [[{ role: 'text', content: 'X'.repeat(200) }]];
        const stream = new MemoryConversationStream(data);

        const tokeniser = new CharTokeniser(300);
        await tokeniser.train([stream]);
        tokeniser.datasetID = 'test-dataset-too-large';

        await expect(
            tokensFromStreams([stream], tokeniser, {
                noOPFS: true,
                shardSize: 32,
            })
        ).rejects.toThrow(/estimated tokens|too small/i);
    });
});
