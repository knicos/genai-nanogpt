import { Tensor, tidy } from '@tensorflow/tfjs-core';
import type { Conversation, ITokeniser } from '../tokeniser/type';
import { Dataset, generator } from '@tensorflow/tfjs-data';
import { Task } from '@base/training/tasks/Task';

export function buildSFTExample(
    conversation: Conversation[],
    ignoreIndex: number,
    tokenizer: ITokeniser,
    blockSize: number
): { xs: Int32Array; ys: Int32Array } | null {
    const tokens: number[] = [tokenizer.bosToken];
    const mask: boolean[] = [false];

    const roleToStart = {
        user: tokenizer.getSpecialTokenIndex('<|user_start|>'),
        assistant: tokenizer.getSpecialTokenIndex('<|assistant_start|>'),
        system: tokenizer.getSpecialTokenIndex('<|system_start|>'),
    } as const;

    const roleToEnd = {
        user: tokenizer.getSpecialTokenIndex('<|user_end|>'),
        assistant: tokenizer.getSpecialTokenIndex('<|assistant_end|>'),
        system: tokenizer.getSpecialTokenIndex('<|system_end|>'),
    } as const;

    for (const fragment of conversation) {
        const start = roleToStart[fragment.role];
        const end = roleToEnd[fragment.role];
        if (start == null || end == null) {
            throw new Error(`Missing special tokens for role: ${fragment.role}`);
        }

        tokens.push(start);
        mask.push(false);

        const contentTokens = tokenizer.encode(fragment.content);
        for (const t of contentTokens) {
            tokens.push(t);
            const isSpecial = tokenizer.isSpecialToken(t);
            const isAssistant = fragment.role === 'assistant';
            mask.push(isAssistant && !isSpecial);
        }

        tokens.push(end);
        mask.push(false);
    }

    tokens.push(tokenizer.eosToken);
    mask.push(false);

    // Ensure length is blockSize + 1 for xs/ys shift
    const targetLen = blockSize + 1;
    if (tokens.length < targetLen) {
        const padCount = targetLen - tokens.length;
        const padToken = tokenizer.getSpecialTokenIndex('<pad>')!;
        for (let i = 0; i < padCount; i++) {
            tokens.push(padToken);
            mask.push(false);
        }
    } else if (tokens.length > targetLen) {
        tokens.length = targetLen;
        mask.length = targetLen;
    }

    const xs = new Int32Array(tokens.slice(0, blockSize));
    const ysRaw = tokens.slice(1, blockSize + 1);
    const maskShifted = mask.slice(1, blockSize + 1);

    const ys = new Int32Array(ysRaw.length);
    let hasUnmasked = false;
    for (let i = 0; i < ysRaw.length; i++) {
        const value = maskShifted[i] ? ysRaw[i] : ignoreIndex;
        ys[i] = value;
        if (value !== ignoreIndex) {
            hasUnmasked = true;
        }
    }

    return hasUnmasked ? { xs, ys } : null;
}

// Training data utilities using TensorFlow.js Dataset API
export class SFTDatasetBuilder {
    public tokenizer: ITokeniser;
    public blockSize: number;

    constructor(tokenizer: ITokeniser, blockSize = 128) {
        this.tokenizer = tokenizer;
        this.blockSize = blockSize;
    }

    /**
     * Create SFT dataset from structured conversations.
     * - Always starts from the beginning of each conversation.
     * - Pads with eosToken and masks padding.
     * - Masks non-assistant tokens in labels with ignoreIndex (default -100).
     */
    public async createSFTDataset(
        conversations: Task[],
        batchSize = 32,
        ignoreIndex = -100
    ): Promise<Dataset<{ xs: Tensor; ys: Tensor }>> {
        if (!conversations.length) {
            throw new Error('No conversations provided.');
        }

        // const examples = conversations.map((conv) => this.buildSFTExample(conv, ignoreIndex));
        const tokeniser = this.tokenizer;
        const blockSize = this.blockSize;

        const gen = function* () {
            while (true) {
                const taskI = Math.floor(Math.random() * conversations.length);
                const task = conversations[taskI];
                const conversation = task.getRandomConversation();
                const example = buildSFTExample(conversation, ignoreIndex, tokeniser, blockSize);
                if (example) {
                    yield example;
                }
            }
        };

        return generator(gen)
            .batch(batchSize)
            .map((batch) => {
                const batchData = batch as { xs: Tensor; ys: Tensor };
                return tidy(() => ({
                    xs: batchData.xs.cast('int32'),
                    ys: batchData.ys.cast('int32'),
                }));
            })
            .prefetch(2);
    }
}
