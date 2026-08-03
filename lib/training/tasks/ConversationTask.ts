import { Conversation, ConversationStream, ConversationCursor, ITokeniser } from '@base/main';
import { Task } from './Task';

export default class ConversationTask extends Task {
    private streams: ConversationStream[];
    private streamIndex = 0;
    private currentCursor: ConversationCursor | null = null;

    get length(): number {
        return this.streams.length;
    }

    constructor(conversations: ConversationStream[]) {
        super();
        this.streams = conversations;
    }

    hasMoreConversations(): boolean {
        return this.streamIndex < this.streams.length;
    }

    async nextConversation(): Promise<Conversation[] | null> {
        if (this.streamIndex < this.streams.length) {
            if (!this.currentCursor) {
                this.currentCursor = this.streams[this.streamIndex].cursor();
            }
            const conv = await this.currentCursor.next();
            if (conv) {
                return conv;
            } else {
                this.streamIndex++;
                this.currentCursor = null;
                return this.nextConversation();
            }
        }
        return null;
    }

    nextTokens(tokeniser: ITokeniser): Promise<number[] | null>;
    nextTokens(tokeniser: ITokeniser, masking: boolean): Promise<{ tokens: number[]; mask: boolean[] } | null>;
    async nextTokens(
        tokeniser: ITokeniser,
        masking?: boolean
    ): Promise<number[] | { tokens: number[]; mask: boolean[] } | null> {
        const conv = await this.nextConversation();
        if (!conv) {
            return null;
        }
        const tokens = tokeniser.encodeConversation(conv, false, masking);
        return tokens;
    }

    async estimateTokens(tokeniser: ITokeniser): Promise<number> {
        const convo = await this.streams[0].cursor().next();
        if (!convo) {
            return 0;
        }
        return tokeniser.encodeConversation(convo).length * this.length;
    }
}
