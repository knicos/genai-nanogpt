import { Conversation, ITokeniser, Roles } from './type';
import EE from 'eventemitter3';

export const SPECIALS = [
    '<eos>',
    '<bos>',
    '',
    '<|user_start|>',
    '<|user_end|>',
    '<|assistant_start|>',
    '<|assistant_end|>',
    '<|system_start|>',
    '<|system_end|>',
];

export default abstract class BaseTokeniser extends EE<'trainStatus'> implements ITokeniser {
    protected specialTokens: Map<string, number> = new Map();
    protected specialTokenSet: Set<number> = new Set();

    abstract vocabSize: number;
    abstract eosToken: number;
    abstract bosToken: number;
    abstract trained: boolean;

    abstract addToken(token: string, index?: number): number;

    public isSpecialToken(index: number): boolean {
        return this.specialTokenSet.has(index);
    }

    protected addSpecialTokens() {
        SPECIALS.forEach((token, index) => {
            this.addToken(token, index);
            this.specialTokens.set(token, index);
            this.specialTokenSet.add(index);
        });
    }

    protected addSpecialToken(token: string, index: number) {
        this.specialTokens.set(token, index);
        this.specialTokenSet.add(index);
    }

    abstract train(text: string[]): Promise<number>;
    abstract getVocab(): string[];
    abstract getMerges(): [string, string][];
    abstract destroy(): void;
    abstract encode(text: string): number[];

    encodeSequence(text: string): number[] {
        const tokens = this.encode(text);
        return [this.bosToken, ...tokens, this.eosToken];
    }

    encodeConversation(conversation: Conversation[], completion?: boolean): number[] {
        const resultTokens: number[][] = [[this.bosToken]];

        const startTokens = [
            this.getSpecialTokenIndex('<|user_start|>')!,
            this.getSpecialTokenIndex('<|assistant_start|>')!,
            this.getSpecialTokenIndex('<|system_start|>')!,
        ];
        const endTokens = [
            this.getSpecialTokenIndex('<|user_end|>')!,
            this.getSpecialTokenIndex('<|assistant_end|>')!,
            this.getSpecialTokenIndex('<|system_end|>')!,
        ];

        for (const fragment of conversation) {
            const encodedContent = this.encode(fragment.content);
            switch (fragment.role) {
                case 'user':
                    resultTokens.push([startTokens[0]]);
                    break;
                case 'assistant':
                    resultTokens.push([startTokens[1]]);
                    break;
                case 'system':
                    resultTokens.push([startTokens[2]]);
                    break;
            }
            resultTokens.push(encodedContent);
            switch (fragment.role) {
                case 'user':
                    resultTokens.push([endTokens[0]]);
                    break;
                case 'assistant':
                    resultTokens.push([endTokens[1]]);
                    break;
                case 'system':
                    resultTokens.push([endTokens[2]]);
                    break;
            }
        }
        const tokens = resultTokens.flat();

        if (completion) {
            tokens.push(startTokens[1]); // Assistant start token for completion
        } else {
            tokens.push(this.eosToken);
        }

        return tokens;
    }

    abstract decode(tokens: number[]): string;

    decodeConversation(tokens: number[]): Conversation[] {
        const conversation: Conversation[] = [];

        let index = 0;
        while (index < tokens.length) {
            const token = tokens[index];
            let role: Roles | null = null;

            if (token === this.getSpecialTokenIndex('<|user_start|>')) {
                role = 'user';
            } else if (token === this.getSpecialTokenIndex('<|assistant_start|>')) {
                role = 'assistant';
            } else if (token === this.getSpecialTokenIndex('<|system_start|>')) {
                role = 'system';
            }

            if (role) {
                index++;
                const contentTokens: number[] = [];
                while (index < tokens.length && tokens[index] !== this.getSpecialTokenIndex(`<|${role}_end|>`)) {
                    contentTokens.push(tokens[index]);
                    index++;
                }
                const content = this.decode(contentTokens);
                conversation.push({ role, content });
            }
            index++;
        }

        return conversation;
    }

    public getSpecialTokenIndex(token: string): number | undefined {
        return this.specialTokens.get(token);
    }
}
