import { describe, it } from 'vitest';
import BPETokeniser from './bpe';

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
});
