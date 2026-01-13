import { Conversation, ITokeniser } from '@base/main';
import { Task } from './Task';

export default class PretrainingTask extends Task {
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
        const conv: Conversation = {
            role: 'assistant',
            content: this.rawText[this.index],
        };
        this.index++;
        return [conv];
    }

    async estimateTokens(tokeniser: ITokeniser): Promise<number> {
        return (
            (
                await tokeniser.encodeConversation([
                    {
                        role: 'assistant',
                        content: this.rawText[0],
                    },
                ])
            ).length * this.length
        );
    }
}
