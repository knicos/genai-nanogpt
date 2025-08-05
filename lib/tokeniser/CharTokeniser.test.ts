import { describe, it } from 'vitest';
import CharTokeniser from './CharTokeniser';

describe('CharTokeniser Tests', () => {
    it('can decode tokens back to text', async ({ expect }) => {
        const charTokeniser = new CharTokeniser();

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        await charTokeniser.train(textData);

        const tokens = await charTokeniser.tokenise(textData, true);
        const eosTokens = tokens.map((t) => [...t, charTokeniser.eosToken]);
        const decodedText = await charTokeniser.detokenise(eosTokens);

        expect(decodedText.map((t) => t.trim())).toEqual(textData.map((t) => t + '<eos>'));
    });
});
