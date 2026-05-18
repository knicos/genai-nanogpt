import { Conversation, ITokeniser } from '@base/main';
import { Task } from './Task';
import { shuffle } from '../DatasetBuilder';

export default class ConversationTask extends Task {
    private rawConvo: Conversation[][];
    private shuffledIndices: Uint32Array | null = null;
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
        const conv = this.rawConvo[this.shuffledIndices ? this.shuffledIndices[this.index] : this.index];
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

    shuffle() {
        if (!this.shuffledIndices) {
            this.shuffledIndices = new Uint32Array(this.rawConvo.length);
            for (let i = 0; i < this.rawConvo.length; i++) {
                this.shuffledIndices[i] = i;
            }
        }
        shuffle(this.shuffledIndices);
        this.index = 0;
    }

    async estimateTokens(tokeniser: ITokeniser): Promise<number> {
        return (await tokeniser.encodeConversation(this.rawConvo[0])).length * this.length;
    }
}
