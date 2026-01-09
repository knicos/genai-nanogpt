import { describe, it } from 'vitest';
import BPETokeniser from './bpe';
import { Conversation } from './type';

describe('BPE Tokeniser Tests', () => {
    it('token per word if possible', async ({ expect }) => {
        const bpe = new BPETokeniser(100);

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        await bpe.train(textData);

        const tokens = await bpe.tokenise(textData);
        expect(tokens).toEqual([
            ['hello', ' world'],
            ['this', ' is', ' a', ' test'],
            ['hello', ' again'],
            ['test', ' the', ' tokenizer'],
        ]);
    });

    it('token per character', async ({ expect }) => {
        const bpe = new BPETokeniser(5);

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        await bpe.train(textData);

        const tokens = await bpe.tokenise(textData);
        expect(tokens).toEqual([
            ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'],
            ['t', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 't', 'e', 's', 't'],
            ['h', 'e', 'l', 'l', 'o', ' ', 'a', 'g', 'a', 'i', 'n'],
            ['t', 'e', 's', 't', ' ', 't', 'h', 'e', ' ', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'e', 'r'],
        ]);
    });

    it('handles an unknown token', async ({ expect }) => {
        const bpe = new BPETokeniser(100);

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        await bpe.train(textData);

        const tokens = await bpe.tokenise(['@']);
        expect(tokens).toEqual([['']]);
    });

    it('handles corrupt data', async ({ expect }) => {
        const bpe = new BPETokeniser(100);

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        await bpe.train(textData);

        // Generate random noise string
        const noise = Array.from({ length: 100 }, () => String.fromCharCode(Math.floor(Math.random() * 256))).join('');

        const tokens = await bpe.tokenise([noise]);
        expect(tokens[0]).toHaveLength(100);
        expect(tokens[0][0]).toEqual('');
    });

    it('handles an unknown token when numeric', async ({ expect }) => {
        const bpe = new BPETokeniser(100);

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        await bpe.train(textData);

        const tokens = await bpe.tokenise(['@'], true);
        expect(tokens).toEqual([[bpe.unkToken]]);
    });

    it('merges white space', async ({ expect }) => {
        const bpe = new BPETokeniser(40);

        const textData = ['    hello', '    is a test', '    hello again'];

        await bpe.train(textData);

        const vocab = bpe.getVocab();
        expect(vocab).toContain('   ');
    });

    it('merges repeated punctuation', async ({ expect }) => {
        const bpe = new BPETokeniser(40);

        const textData = ['hello!!!', 'this is a test...', 'hello again!!!', '\t\t\twow'];

        await bpe.train(textData);

        const vocab = bpe.getVocab();
        expect(vocab).toContain('!!!');
        expect(vocab).toContain('...');
        expect(vocab).toContain('\t\t\t');
    });

    it('can decode tokens back to text', async ({ expect }) => {
        const bpe = new BPETokeniser(100);

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        await bpe.train(textData);

        const tokens = await bpe.tokenise(textData, true);
        const eosTokens = tokens.map((t) => [...t, bpe.eosToken]);
        const decodedText = await bpe.detokenise(eosTokens);

        expect(decodedText.map((t) => t.trim())).toEqual(textData.map((t) => t + '<eos>'));
    });

    it('can encode and decode a conversation', async ({ expect }) => {
        const bpeTokeniser = new BPETokeniser(100);

        const conversation: Conversation[] = [
            { role: 'user', content: 'Hello, how are you?' },
            { role: 'assistant', content: 'I am fine, thank you!' },
            { role: 'system', content: 'This is a system message.' },
        ];

        await bpeTokeniser.train(conversation.map((c) => c.content));

        const encoded = await bpeTokeniser.encodeConversation(conversation);
        const decoded = await bpeTokeniser.decodeConversation(encoded);

        expect(decoded).toEqual(conversation);
    });
});
