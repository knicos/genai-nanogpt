import { describe, it } from 'vitest';
import { tokensFromTasks } from './Task';
import { CharTokeniser, Conversation } from '@base/main';
import ConversationTask from './ConversationTask';

describe('Task', () => {
    it('can generate tokens from multiple tasks', async ({ expect }) => {
        const data1: Conversation[][] = [
            [
                { role: 'text', content: 'Hello world.' },
                { role: 'text', content: 'How are you?' },
            ],
        ];
        const data2: Conversation[][] = [
            [
                { role: 'text', content: 'This is a test.' },
                { role: 'text', content: 'Testing 123.' },
            ],
        ];
        const task1 = new ConversationTask(data1);
        const task2 = new ConversationTask(data2);

        const tasks = [task1, task2];

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train(data1.concat(data2));
        const tokens = await tokensFromTasks(tasks, tokeniser);

        expect(tokens.length).toBeGreaterThan(data1.length + data2.length); // Should be more tokens than sentences
        const decodedText = tokeniser.decodeConversation(tokens);

        expect(decodedText[0].content).toContain('Hello world.How are you?');
        expect(decodedText[0].role).toBe('text');
        expect(decodedText[1].content).toContain('This is a test.Testing 123.');
        expect(decodedText[1].role).toBe('text');
    });

    it('can generate a mask', async ({ expect }) => {
        const data1: Conversation[][] = [
            [
                { role: 'user', content: 'Hello world.' },
                { role: 'assistant', content: 'How are you?' },
            ],
        ];
        const data2: Conversation[][] = [
            [
                { role: 'user', content: 'This is a test.' },
                { role: 'assistant', content: 'Testing 123.' },
            ],
        ];
        const task1 = new ConversationTask(data1);
        const task2 = new ConversationTask(data2);

        const tasks = [task1, task2];

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train(data1.concat(data2));
        const tokens = await tokensFromTasks(tasks, tokeniser, undefined, true);

        const decodedText = tokeniser.decodeConversation(tokens.tokens);

        console.log('Mask:', tokens.mask);

        expect(tokens.mask.length).toBe(tokens.tokens.length);

        expect(decodedText[0].content).toContain('Hello world.');
        expect(decodedText[0].role).toBe('user');
        expect(decodedText[1].content).toContain('How are you?');
        expect(decodedText[1].role).toBe('assistant');
    });

    it('can handle re-expansion of array when large token count', async ({ expect }) => {
        const data1: Conversation[][] = [[{ role: 'text', content: 'short first sentence' }]];

        for (let i = 0; i < 50; i++) {
            // Create a large string
            data1.push([{ role: 'text', content: `This is sentence number ${i}. ` + 'A'.repeat(100) }]);
        }
        const task1 = new ConversationTask(data1);

        const tasks = [task1];

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train(data1);

        const tokens = await tokensFromTasks(tasks, tokeniser);

        expect(tokens.length).toBeGreaterThan(data1.length); // Should be more tokens than sentences
        const decodedText = tokeniser.decodeConversation(tokens);

        for (let i = 0; i < data1.length; i++) {
            expect(decodedText[i].content).toContain(data1[i].map((c) => c.content).join(''));
            expect(decodedText[i].role).toBe('text');
        }
    });

    it('can generate tokens from multiple different tasks', async ({ expect }) => {
        const data1: Conversation[][] = [
            [
                { role: 'text', content: 'Hello world.' },
                { role: 'text', content: 'How are you?' },
            ],
        ];
        const data2: Conversation[][] = [
            [
                { role: 'text', content: 'This is a test. You now must complete the sentence.' },
                { role: 'text', content: 'Testing 123. 123 Testing.' },
            ],
        ];
        const task1 = new ConversationTask(data1);
        const task2 = new ConversationTask(data2);

        const tasks = [task1, task2];

        const tokeniser = new CharTokeniser(200);
        await tokeniser.train(data1.concat(data2));

        const tokens = await tokensFromTasks(tasks, tokeniser);

        expect(tokens.length).toBeGreaterThan(data1.length + data2.length); // Should be more tokens than sentences
        const decodedText = await tokeniser.decode(tokens);

        expect(decodedText).toContain(
            '<bos>Hello world.How are you?<eos><bos>This is a test. You now must complete the sentence.Testing 123. 123 Testing.<eos>'
        );
    });
});
