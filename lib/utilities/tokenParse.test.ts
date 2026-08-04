import { describe, it } from 'vitest';
import tokenParse from './tokenParse';

describe('Token parsing', () => {
    it('should parse tokens correctly', ({ expect }) => {
        // Example test case for token parsing
        const input = 'Hello, world!';
        const expectedTokens = ['Hello', ',', ' world', '!'];
        const parsedTokens = tokenParse(input);
        expect(parsedTokens).toEqual(expectedTokens);
    });

    it('should join punctuation and symbols correctly', ({ expect }) => {
        const input = 'Hello!!! How are you???';
        const expectedTokens = ['Hello', '!!!', ' How', ' are', ' you', '???'];
        const parsedTokens = tokenParse(input);
        expect(parsedTokens).toEqual(expectedTokens);
    });

    it('should handle multiple spaces correctly', ({ expect }) => {
        const input = 'Hello   world';
        const expectedTokens = ['Hello  ', ' world'];
        const parsedTokens = tokenParse(input);
        expect(parsedTokens).toEqual(expectedTokens);
    });
});
