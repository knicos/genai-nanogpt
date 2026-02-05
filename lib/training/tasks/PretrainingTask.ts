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

    nextTokens(tokeniser: ITokeniser): number[] | null {
        if (this.index >= this.rawText.length) {
            return null;
        }
        const tokens = tokeniser.encodeSequence(this.rawText[this.index]);
        this.index++;
        return tokens;
    }

    getRandomConversation(): Conversation[] {
        const i = Math.floor(Math.random() * this.rawText.length);
        return [
            {
                role: 'assistant',
                content: this.rawText[i],
            },
        ];
    }

    getRandomTokens(tokeniser: ITokeniser): number[] {
        const i = Math.floor(Math.random() * this.rawText.length);
        return tokeniser.encodeSequence(this.rawText[i]);
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
