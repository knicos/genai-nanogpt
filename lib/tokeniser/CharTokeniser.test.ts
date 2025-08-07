import { describe, it } from 'vitest';
import CharTokeniser from './CharTokeniser';

describe('CharTokeniser Tests', () => {
    it('can decode tokens back to text', async ({ expect }) => {
        const charTokeniser = new CharTokeniser(100);

        const textData = ['hello world', 'this is a test', 'hello again', 'test the tokenizer'];

        await charTokeniser.train(textData);

        const tokens = await charTokeniser.tokenise(textData, true);
        const eosTokens = tokens.map((t) => [...t, charTokeniser.eosToken]);
        const decodedText = await charTokeniser.detokenise(eosTokens);

        expect(decodedText.map((t) => t.trim())).toEqual(textData.map((t) => t + '<eos>'));
    });

    it('pads the vocabulary correctly', async ({ expect }) => {
        const charTokeniser = new CharTokeniser(10);

        const textData = ['short', 'sort'];

        await charTokeniser.train(textData);

        expect(charTokeniser.vocabSize).toBe(10);
        expect(charTokeniser.vocab).toContain('<pad>');
        expect(charTokeniser.vocab).toContain('<eos>');
        expect(charTokeniser.vocab).toHaveLength(10);
    });

    it('ignores least common characters when vocab size is exceeded', async ({ expect }) => {
        const charTokeniser = new CharTokeniser(5);

        const textData = ['a', 'b', 'c', 'c', 'a', 'b', 'd', 'd', 'e', 'e', 'f', 'g'];

        await charTokeniser.train(textData);

        console.log('Vocab:', charTokeniser.vocab);

        expect(charTokeniser.vocabSize).toBe(5);
        expect(charTokeniser.vocab).not.toContain('f');
        expect(charTokeniser.vocab).not.toContain('g');
        expect(charTokeniser.vocab).toContain('<eos>');
        expect(charTokeniser.vocab).toHaveLength(5);
    });

    it('replaces unknown characters with <unk>', async ({ expect }) => {
        const charTokeniser = new CharTokeniser(5);

        const textData = ['a', 'b', 'c', 'c', 'a', 'b', 'd', 'd', 'e', 'e', 'f', 'g'];

        await charTokeniser.train(textData);

        const tokens = (await charTokeniser.tokenise(textData)).flat();

        expect(tokens).toContain('<unk>');
    });
});
