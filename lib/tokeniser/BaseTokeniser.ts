import { ConversationStream } from '@base/data/stream';
import { Conversation, ITokeniser, Roles } from './type';
import EE from 'eventemitter3';

export const SPECIALS = [
    '<eos>',
    '<bos>',
    '',
    '<pad>',
    '<|user_start|>',
    '<|user_end|>',
    '<|assistant_start|>',
    '<|assistant_end|>',
    '<|system_start|>',
    '<|system_end|>',
];

export default abstract class BaseTokeniser extends EE<'trainStatus'> implements ITokeniser {
    id = 'untrained';
    datasetID?: string;
    protected specialTokens = new Map<string, number>();
    protected specialTokenSet = new Set<number>();

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

    public generateID() {
        const vocab = this.getVocab();
        let h1 = 0x811c9dc5; // FNV-like
        let h2 = 0x9e3779b9; // second stream

        if (vocab.length === 0) {
            this.id = 'untrained';
            return;
        }

        for (let i = 0; i < vocab.length; i++) {
            const token = vocab[i];
            h1 ^= token.length;
            h1 = Math.imul(h1, 0x01000193);

            h2 ^= i;
            h2 = Math.imul(h2, 0x85ebca6b);

            for (let j = 0; j < token.length; j++) {
                const c = token.charCodeAt(j);

                h1 ^= c;
                h1 = Math.imul(h1, 0x01000193);

                h2 ^= c;
                h2 = Math.imul(h2, 0xc2b2ae35);
            }
        }

        const a = (h1 >>> 0).toString(36);
        const b = (h2 >>> 0).toString(36);
        this.id = 'tokeniser_' + a + '_' + b;
    }

    abstract train(text: ConversationStream[], cb?: (vocab: number) => void, datasetID?: string): Promise<number>;
    abstract getVocab(): string[];
    abstract getMerges(): [string, string][];
    abstract destroy(): void;
    abstract encode(text: string): number[];

    encodeSequence(text: string): number[] {
        const tokens = this.encode(text);
        return [this.bosToken, ...tokens, this.eosToken];
    }

    encodeAsSequence(conversation: Conversation[], completion?: boolean): number[] {
        const tokens = conversation.flatMap((fragment) => {
            const encodedContent = this.encode(fragment.content);
            return encodedContent;
        });
        return completion
            ? [this.bosToken, ...tokens, this.eosToken, this.bosToken]
            : [this.bosToken, ...tokens, this.eosToken];
    }

    encodeConversation(conversation: Conversation[], completion?: boolean): number[];
    encodeConversation(
        conversation: Conversation[],
        completion: boolean,
        masking: boolean
    ): { tokens: number[]; mask: boolean[] };
    encodeConversation(
        conversation: Conversation[],
        completion?: boolean,
        masking?: boolean
    ): number[] | { tokens: number[]; mask: boolean[] } {
        const resultTokens: number[][] = [[this.bosToken]];
        let mask: boolean[][] | undefined = undefined;
        if (masking) {
            mask = [[false]]; // mask BOS token
        }

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
            let maskContent = false;
            const encodedContent = this.encode(fragment.content);
            switch (fragment.role) {
                case 'user':
                    resultTokens.push([startTokens[0]]);
                    maskContent = true;
                    break;
                case 'assistant':
                    resultTokens.push([startTokens[1]]);
                    break;
                case 'system':
                    resultTokens.push([startTokens[2]]);
                    maskContent = true;
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

            if (masking && mask && maskContent) {
                // Mask all content tokens, but not special tokens
                mask.push([false]); // for start token
                mask.push(encodedContent.map(() => false));
                mask.push([false]); // for end token
            } else if (masking && mask) {
                // Unmask all tokens for this fragment
                mask.push([false]); // for start token
                mask.push(encodedContent.map(() => true));
                mask.push([true]); // for end token
            }
        }
        const tokens = resultTokens.flat();

        if (completion) {
            tokens.push(startTokens[1]); // Assistant start token for completion
            if (masking && mask) {
                mask.push([false]);
            }
        } else {
            tokens.push(this.eosToken);
            if (masking && mask) {
                mask.push([true]);
            }
        }

        return masking && mask ? { tokens, mask: mask.flat() } : tokens;
    }

    abstract decode(tokens: number[]): string;

    decodeConversation(tokens: number[] | Uint16Array): Conversation[] {
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
            } else if (token === this.bosToken) {
                // skip
            } else if (token === this.eosToken) {
                role = null;
            } else {
                role = 'text';
                index--; // Step back to include this token in content
            }

            if (role) {
                index++;
                const contentTokens: number[] = [];
                while (
                    index < tokens.length &&
                    tokens[index] !== this.getSpecialTokenIndex(`<|${role}_end|>`) &&
                    tokens[index] !== this.eosToken
                ) {
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
