import { describe, it } from 'vitest';
import NodeTokeniser from './NodeTokeniser';

describe('NodeTokeniser Tests', () => {
    it('can decode tokens back to text', async ({ expect }) => {
        const bpe = new NodeTokeniser();

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        bpe.train(textData, 100);

        const tokens = await bpe.tokenise(textData, true);
        const eosTokens = tokens.map((t) => [...t, bpe.eosToken]);
        const decodedText = await bpe.detokenise(eosTokens);

        expect(decodedText.map((t) => t.trim())).toEqual(textData.map((t) => t + '<eos>'));
    });
});
