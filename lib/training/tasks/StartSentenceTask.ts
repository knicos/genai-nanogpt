import { Conversation, ITokeniser } from '@base/main';
import { Task } from './Task';

// Uses the first sentence as a user prompt to start the conversation.
export default class StartSentenceTask extends Task {
    private rawText: string[];
    private index = 0;

    get length(): number {
        return this.rawText.length;
    }

    constructor(texts: string[]) {
        super();
        this.rawText = texts;
    }

    hasMoreConversations(): boolean {
        return this.index < this.rawText.length;
    }

    nextConversation(): Conversation[] | null {
        if (this.index >= this.rawText.length) {
            return null;
        }

        const text = this.rawText[this.index];
        this.index++;

        return this.conversationFromString(text);
    }

    private conversationFromString(text: string): Conversation[] {
        const endOfFirstSentence = text.indexOf('.');

        if (endOfFirstSentence === -1) {
            const conv: Conversation = {
                role: 'assistant',
                content: this.rawText[this.index],
            };

            return [conv];
        }

        const conv: Conversation[] = [
            {
                role: 'user',
                content: text.slice(0, endOfFirstSentence + 1).trim(),
            },
            {
                role: 'assistant',
                content: text.slice(endOfFirstSentence + 1).trim(),
            },
        ];

        return conv;
    }

    async estimateTokens(tokeniser: ITokeniser): Promise<number> {
        return (await tokeniser.encodeConversation(this.conversationFromString(this.rawText[0]))).length * this.length;
    }
}
