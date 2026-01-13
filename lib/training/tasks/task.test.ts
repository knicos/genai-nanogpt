import { describe, it } from 'vitest';
import PretrainingTask from './PretrainingTask';
import { tokensFromTasks } from './Task';
import { CharTokeniser } from '@base/main';
import StartSentenceTask from './StartSentenceTask';

describe('Task', () => {
    it('can generate tokens from multiple tasks', async ({ expect }) => {
        const data1 = ['Hello world.', 'How are you?'];
        const data2 = ['This is a test.', 'Testing 123.'];
        const task1 = new PretrainingTask(data1);
        const task2 = new PretrainingTask(data2);

        const tasks = [task1, task2];

        const tokeniser = new CharTokeniser(200);
        tokeniser.train(data1.concat(data2));

        const tokens = await tokensFromTasks(tasks, tokeniser);

        expect(tokens.length).toBeGreaterThan(data1.length + data2.length); // Should be more tokens than sentences
        const decodedText = await tokeniser.decode(tokens);

        for (const sentence of data1.concat(data2)) {
            expect(decodedText).toContain(sentence);
        }
    });

    it('can handle re-expansion of array when large token count', async ({ expect }) => {
        const data1: string[] = ['short first sentence'];

        for (let i = 0; i < 50; i++) {
            // Create a large string
            data1.push(`This is sentence number ${i}. ` + 'A'.repeat(100));
        }
        const task1 = new PretrainingTask(data1);

        const tasks = [task1];

        const tokeniser = new CharTokeniser(200);
        tokeniser.train(data1);

        const tokens = await tokensFromTasks(tasks, tokeniser);

        expect(tokens.length).toBeGreaterThan(data1.length); // Should be more tokens than sentences
        const decodedText = await tokeniser.decode(tokens);

        for (const sentence of data1) {
            expect(decodedText).toContain(sentence);
        }
    });

    it('can generate tokens from multiple different tasks', async ({ expect }) => {
        const data1 = ['Hello world.', 'How are you?'];
        const data2 = ['This is a test. You now must complete the sentence.', 'Testing 123. 123 Testing.'];
        const task1 = new PretrainingTask(data1);
        const task2 = new StartSentenceTask(data2);

        const tasks = [task1, task2];

        const tokeniser = new CharTokeniser(200);
        tokeniser.train(data1.concat(data2));

        const tokens = await tokensFromTasks(tasks, tokeniser);

        expect(tokens.length).toBeGreaterThan(data1.length + data2.length); // Should be more tokens than sentences
        const decodedText = await tokeniser.decode(tokens);
        expect(decodedText).toContain(
            '<bos><|user_start|>This is a test.<|user_end|><|assistant_start|>You now must complete the sentence.<|assistant_end|><eos>'
        );
    });
});
