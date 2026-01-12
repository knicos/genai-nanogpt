import { describe, it } from 'vitest';
import CharTokeniser from './CharTokeniser';
import { SPECIALS } from './BaseTokeniser';
import { Conversation } from './type';

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
        const charTokeniser = new CharTokeniser(20);

        const textData = ['short', 'sort'];

        await charTokeniser.train(textData);

        expect(charTokeniser.vocabSize).toBe(20);
        expect(charTokeniser.vocab.length).toBe(20);
        expect(charTokeniser.vocab.filter((token) => token === '')).toHaveLength(20 - (SPECIALS.length + 5 - 1));
        expect(charTokeniser.vocab).toContain('<eos>');
        expect(charTokeniser.vocab).toHaveLength(20);
    });

    it('ignores least common characters when vocab size is exceeded', async ({ expect }) => {
        const charTokeniser = new CharTokeniser(SPECIALS.length + 3);

        const textData = ['a', 'b', 'c', 'c', 'a', 'b', 'd', 'd', 'e', 'e', 'f', 'g'];

        await charTokeniser.train(textData);

        expect(charTokeniser.vocabSize).toBe(SPECIALS.length + 3);
        expect(charTokeniser.vocab).not.toContain('f');
        expect(charTokeniser.vocab).not.toContain('g');
        expect(charTokeniser.vocab).toContain('<eos>');
        expect(charTokeniser.vocab).toHaveLength(SPECIALS.length + 3);
    });

    it('replaces unknown characters with <unk>', async ({ expect }) => {
        const charTokeniser = new CharTokeniser(SPECIALS.length + 3);

        const textData = ['a', 'b', 'c', 'c', 'a', 'b', 'd', 'd', 'e', 'e', 'f', 'g'];

        await charTokeniser.train(textData);

        const tokens = (await charTokeniser.tokenise(textData)).flat();

        expect(tokens).toContain('');
    });

    it('replaces <pad> if train called again', async ({ expect }) => {
        const charTokeniser = new CharTokeniser(30);

        const textData1 = ['hello world', 'hello again'];
        const textData2 = ['short', 'sort'];

        await charTokeniser.train(textData1);

        const vocabAfterFirstTrain = [...charTokeniser.vocab];

        await charTokeniser.train(textData2);

        const vocabAfterSecondTrain = [...charTokeniser.vocab];

        expect(vocabAfterFirstTrain).not.toEqual(vocabAfterSecondTrain);
        expect(vocabAfterSecondTrain).toContain('s');
        expect(vocabAfterSecondTrain).toContain('o');
        expect(vocabAfterSecondTrain).toContain('r');
        expect(vocabAfterSecondTrain).toContain('t');
        expect(vocabAfterSecondTrain).toContain('');
        expect(vocabAfterSecondTrain[2]).toBe('');
        expect(vocabAfterSecondTrain[1]).toBe('<bos>');
        expect(vocabAfterFirstTrain).toHaveLength(30);
        expect(vocabAfterSecondTrain).toHaveLength(30);
    });

    it('can encode and decode a conversation', async ({ expect }) => {
        const charTokeniser = new CharTokeniser(100);

        const conversation: Conversation[] = [
            { role: 'user', content: 'Hello, how are you?' },
            { role: 'assistant', content: 'I am fine, thank you!' },
            { role: 'system', content: 'This is a system message.' },
        ];

        await charTokeniser.train(conversation.map((c) => c.content));

        const encoded = await charTokeniser.encodeConversation(conversation);

        expect(encoded[0]).toBe(charTokeniser.bosToken);
        expect(encoded[encoded.length - 1]).toBe(charTokeniser.eosToken);
        expect(encoded).toContain(charTokeniser.getSpecialTokenIndex('<|user_start|>')!);
        expect(encoded).toContain(charTokeniser.getSpecialTokenIndex('<|assistant_start|>')!);
        expect(encoded).toContain(charTokeniser.getSpecialTokenIndex('<|assistant_end|>')!);

        const decoded = await charTokeniser.decodeConversation(encoded);

        expect(decoded).toEqual(conversation);
    });
});
