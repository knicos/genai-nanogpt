import { describe, it } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import type { Conversation, ITokeniser } from '../tokeniser/type';
import { SFTDatasetBuilder } from './SFTDatasetBuilder';
import ConversationTask from './tasks/ConversationTask';

await tf.setBackend('cpu');

function makeMockTokenizer(): ITokeniser {
    const specials = new Map<string, number>([
        ['<eos>', 0],
        ['<bos>', 1],
        ['', 2],
        ['<pad>', 3],
        ['<|user_start|>', 4],
        ['<|user_end|>', 5],
        ['<|assistant_start|>', 6],
        ['<|assistant_end|>', 7],
        ['<|system_start|>', 8],
        ['<|system_end|>', 9],
    ]);
    const specialSet = new Set<number>(specials.values());

    return {
        vocabSize: 256,
        eosToken: 0,
        bosToken: 1,
        trained: true,
        addToken: () => 0,
        isSpecialToken: (index: number) => specialSet.has(index),
        getSpecialTokenIndex: (token: string) => specials.get(token),
        encode: (text: string) => text.split('').map((c) => c.charCodeAt(0) + 100),
        encodeConversation: async () => [],
        decode: () => '',
        getVocab: () => [],
        getMerges: () => [],
        train: async () => 0,
        destroy: () => undefined,
        encodeSequence: (text: string) => [1, ...text.split('').map((c) => c.charCodeAt(0) + 100), 0],
        decodeConversation: () => [],
    } as unknown as ITokeniser;
}

function buildExpectedExample(
    conversation: Conversation[],
    ignoreIndex: number,
    tokenizer: ITokeniser,
    blockSize: number
) {
    const tokens: number[] = [tokenizer.bosToken];
    const mask: boolean[] = [false];

    const roleToStart = {
        user: tokenizer.getSpecialTokenIndex('<|user_start|>'),
        assistant: tokenizer.getSpecialTokenIndex('<|assistant_start|>'),
        system: tokenizer.getSpecialTokenIndex('<|system_start|>'),
    } as const;

    const roleToEnd = {
        user: tokenizer.getSpecialTokenIndex('<|user_end|>'),
        assistant: tokenizer.getSpecialTokenIndex('<|assistant_end|>'),
        system: tokenizer.getSpecialTokenIndex('<|system_end|>'),
    } as const;

    for (const fragment of conversation) {
        const isAssistant = fragment.role === 'assistant';
        const start = roleToStart[fragment.role]!;
        const end = roleToEnd[fragment.role]!;

        tokens.push(start);
        mask.push(false);

        const contentTokens = tokenizer.encode(fragment.content);
        for (const t of contentTokens) {
            tokens.push(t);
            const isSpecial = tokenizer.isSpecialToken(t);
            mask.push(isAssistant && !isSpecial);
        }

        tokens.push(end);
        mask.push(isAssistant);
    }

    tokens.push(tokenizer.eosToken);
    mask.push(false);

    const targetLen = blockSize + 1;
    if (tokens.length < targetLen) {
        const padCount = targetLen - tokens.length;
        const padToken = tokenizer.getSpecialTokenIndex('<pad>')!;
        for (let i = 0; i < padCount; i++) {
            tokens.push(padToken);
            mask.push(false);
        }
    } else if (tokens.length > targetLen) {
        tokens.length = targetLen;
        mask.length = targetLen;
    }

    const xs = tokens.slice(0, blockSize);
    const ysRaw = tokens.slice(1, blockSize + 1);
    const maskShifted = mask.slice(1, blockSize + 1);
    const ys = ysRaw.map((t, i) => (maskShifted[i] ? t : ignoreIndex));

    return { xs, ys };
}

describe('SFTDatasetBuilder', () => {
    it('creates SFT dataset with masked labels', async ({ expect }) => {
        const tokenizer = makeMockTokenizer();
        const blockSize = 32;
        const ignoreIndex = -100;

        const builder = new SFTDatasetBuilder(tokenizer, blockSize);

        const conversations: Conversation[][] = [
            [
                { role: 'user', content: 'hi' },
                { role: 'assistant', content: 'ok' },
                { role: 'user', content: 'fine' },
                { role: 'assistant', content: 'no' },
            ],
        ];

        const task = new ConversationTask(conversations);

        const dataset = await builder.createSFTDataset([task], 1, ignoreIndex);
        const iterator = await dataset.iterator();
        const firstBatch = await iterator.next();
        const value = firstBatch.value as { xs: tf.Tensor; ys: tf.Tensor };

        expect(value.xs.shape).toEqual([1, blockSize]);
        expect(value.ys.shape).toEqual([1, blockSize]);

        const xs = (await value.xs.array()) as number[][];
        const ys = (await value.ys.array()) as number[][];

        const expected = buildExpectedExample(conversations[0], ignoreIndex, tokenizer, blockSize);

        console.log('Expected xs:', expected.xs);
        console.log('Expected ys:', expected.ys);
        console.log('Actual xs:', xs[0]);
        console.log('Actual ys:', ys[0]);

        expect(xs[0]).toEqual(expected.xs);
        expect(ys[0]).toEqual(expected.ys);
    });
});
