import { describe, it } from 'vitest';
import BPE from './bpe';

describe('BPE Tokeniser Tests', () => {
    it('token per word if possible', async ({ expect }) => {
        const bpe = new BPE();

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        bpe.train(textData, 100);

        const tokens = bpe.tokenise(textData);
        expect(tokens).toEqual([
            ['hello', ' world'],
            ['this', ' is', ' a', ' test'],
            ['hello', ' again'],
            ['test', ' the', ' tokenizer'],
        ]);
    });

    it('token per character', async ({ expect }) => {
        const bpe = new BPE();

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        bpe.train(textData, 5);

        const tokens = bpe.tokenise(textData);
        expect(tokens).toEqual([
            ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'],
            ['t', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 't', 'e', 's', 't'],
            ['h', 'e', 'l', 'l', 'o', ' ', 'a', 'g', 'a', 'i', 'n'],
            ['t', 'e', 's', 't', ' ', 't', 'h', 'e', ' ', 't', 'o', 'k', 'e', 'n', 'i', 'z', 'e', 'r'],
        ]);
    });
});
