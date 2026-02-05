import { Conversation, ITokeniser } from '@base/main';
import { Task } from './Task';

export default class ConversationTask extends Task {
    private rawConvo: Conversation[][];
    private index = 0;

    get length(): number {
        return this.rawConvo.length;
    }

    constructor(conversations: Conversation[][]) {
        super();
        this.rawConvo = conversations;
    }

    hasMoreConversations(): boolean {
        return this.index < this.rawConvo.length;
    }

    nextConversation(): Conversation[] | null {
        if (this.index >= this.rawConvo.length) {
            return null;
        }
        const conv = this.rawConvo[this.index];
        this.index++;
        return conv;
    }

    nextTokens(tokeniser: ITokeniser): number[] | null {
        const conv = this.nextConversation();
        if (!conv) {
            return null;
        }
        const tokens = tokeniser.encodeConversation(conv);
        return tokens;
    }

    getRandomConversation(): Conversation[] {
        const i = Math.floor(Math.random() * this.rawConvo.length);
        return this.rawConvo[i];
    }

    getRandomTokens(tokeniser: ITokeniser): number[] {
        const i = Math.floor(Math.random() * this.rawConvo.length);
        return tokeniser.encodeConversation(this.rawConvo[i]);
    }

    async estimateTokens(tokeniser: ITokeniser): Promise<number> {
        return (await tokeniser.encodeConversation(this.rawConvo[0])).length * this.length;
    }
}
