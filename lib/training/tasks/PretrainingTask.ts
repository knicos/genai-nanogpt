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

    nextTokens(tokeniser: ITokeniser): number[] | null;
    nextTokens(tokeniser: ITokeniser, masking: boolean): { tokens: number[]; mask: boolean[] } | null;
    nextTokens(tokeniser: ITokeniser, masking?: boolean): number[] | { tokens: number[]; mask: boolean[] } | null {
        if (this.index >= this.rawText.length) {
            return null;
        }
        const tokens = tokeniser.encodeSequence(this.rawText[this.index]);
        this.index++;
        if (masking) {
            const mask = new Array(tokens.length).fill(true);
            return { tokens, mask };
        }
        return tokens;
    }

    shuffle() {
        // NOP
        this.index = 0;
    }

    async estimateTokens(tokeniser: ITokeniser): Promise<number> {
        return (
            tokeniser.encodeConversation([
                {
                    role: 'assistant',
                    content: this.rawText[0],
                },
            ]).length * this.length
        );
    }
}
