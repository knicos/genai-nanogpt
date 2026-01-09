import { Conversation, ITokeniser, Roles } from './type';
import EE from 'eventemitter3';

export const SPECIALS = [
    '<eos>',
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

    abstract vocabSize: number;
    abstract eosToken: number;
    abstract trained: boolean;

    abstract addToken(token: string, index?: number): number;

    protected addSpecialTokens() {
        SPECIALS.forEach((token, index) => {
            this.addToken(token, index);
            this.specialTokens.set(token, index);
        });
    }

    protected addSpecialToken(token: string, index: number) {
        this.specialTokens.set(token, index);
    }

    abstract train(text: string[]): Promise<number>;
    abstract tokenise(text: string[], numeric?: boolean): Promise<string[][] | number[][]>;
    abstract detokenise(tokens: string[][] | number[][]): Promise<string[]>;
    abstract getVocab(): string[];
    abstract getMerges(): Promise<[string, string][]>;
    abstract destroy(): void;
    abstract encode(text: string): Promise<number[]>;

    async encodeConversation(conversation: Conversation[], completion?: boolean): Promise<number[]> {
        const resultTokens: number[][] = [];

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
            const encodedContent = await this.encode(fragment.content);
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
        }

        return tokens;
    }

    abstract decode(tokens: number[]): Promise<string>;

    async decodeConversation(tokens: number[]): Promise<Conversation[]> {
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
                const content = await this.decode(contentTokens);
                conversation.push({ role, content });
            }
            index++;
        }

        return conversation;
    }

    protected getSpecialTokenIndex(token: string): number | undefined {
        return this.specialTokens.get(token);
    }
}
