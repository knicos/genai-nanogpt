import { describe, it, vi } from 'vitest';
import { DatasetBuilder, flattenTokens } from './DatasetBuilder';
import { Conversation, ITokeniser } from '../tokeniser/type';
import { createTrainValidationSplit } from './validation';

describe('Validation split', () => {
    it('should split with the correct size', async ({ expect }) => {
        const mockTokenizer = {
            vocabSize: 256,
            encodeConversation: vi.fn(async (conversation: Conversation[]) =>
                conversation.map((msg) => msg.content.split('').map((c: string) => c.charCodeAt(0))).flat()
            ),
        } as unknown as ITokeniser;
        const blockSize = 1;

        // Create instance of DatasetBuilder
        const datasetBuilder = new DatasetBuilder(mockTokenizer, blockSize);

        const textData: Conversation[] = [{ role: 'user', content: 'hello world hello world hello world hello world' }];
        const allTokens = new Uint16Array(await flattenTokens([textData], mockTokenizer));

        const { trainState, validationState, size } = await createTrainValidationSplit(
            allTokens,
            mockTokenizer,
            datasetBuilder,
            2,
            0.25
        );

        expect(size).toBe(allTokens.length);
        expect(trainState.shuffledIndexes.length).toBeGreaterThan(validationState.shuffledIndexes.length);
        expect(trainState.shuffledIndexes.length).toBeGreaterThan(Math.floor(allTokens.length * 0.7));
        expect(validationState.shuffledIndexes.length).toBeLessThan(Math.floor(allTokens.length * 0.3));

        // Confirm that no indexes overlap between train and validation
        const trainSet = new Set(trainState.shuffledIndexes);
        let hasOverlap = false;
        for (const idx of validationState.shuffledIndexes) {
            if (trainSet.has(idx)) {
                hasOverlap = true;
                break;
            }
        }
        expect(hasOverlap).toBe(false);
    });
});
