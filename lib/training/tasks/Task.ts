import { Conversation, ITokeniser } from '@base/main';
import { yieldIfNeeded } from '@base/utilities/yielder';

export abstract class Task {
    abstract get length(): number;
    abstract hasMoreConversations(): boolean;
    abstract nextConversation(): Promise<Conversation[] | null>;

    abstract nextTokens(tokeniser: ITokeniser): Promise<number[] | null>;
    abstract nextTokens(tokeniser: ITokeniser, masking: boolean): Promise<{ tokens: number[]; mask: boolean[] } | null>;
    abstract nextTokens(
        tokeniser: ITokeniser,
        masking?: boolean
    ): Promise<number[] | { tokens: number[]; mask: boolean[] } | null>;

    //abstract estimateTokens(tokeniser: ITokeniser): Promise<number>;
    //abstract shuffle(): void;
}

async function roundRobinData(
    tasks: Task[],
    allTokens: Uint16Array[],
    tokenizer: ITokeniser,
    state: { offset: number; total: number },
    estimatedTokens: number,
    mask?: Uint8Array[]
) {
    // Step through each task in round-robin fashion
    for (const task of tasks) {
        const tokens = await task.nextTokens(tokenizer, mask ? true : undefined);
        if (tokens) {
            const tokenArray = Array.isArray(tokens) ? tokens : tokens.tokens;
            state.total += tokenArray.length;

            const currentTokens = allTokens[allTokens.length - 1];
            const currentMask = mask ? mask[mask.length - 1] : null;

            // Fill existing array and/or create a new array for remaining tokens if needed
            if (state.offset + tokenArray.length > currentTokens.length) {
                const remainingSpace = currentTokens.length - state.offset;
                currentTokens.set(tokenArray.slice(0, remainingSpace), state.offset);
                const neededSize = tokenArray.length - remainingSpace;
                if (neededSize > estimatedTokens) {
                    throw new Error(
                        `Estimated tokens (${estimatedTokens}) is too small for the next batch of tokens (${neededSize}).`
                    );
                }
                const newArray = new Uint16Array(estimatedTokens);
                newArray.set(tokenArray.slice(remainingSpace), 0);
                allTokens.push(newArray);

                if (mask && currentMask && !Array.isArray(tokens)) {
                    currentMask.set(
                        tokens.mask.slice(0, remainingSpace).map((m) => (m ? 1 : 0)),
                        state.offset
                    );
                    const newMask = new Uint8Array(newArray.length);
                    newMask.set(
                        tokens.mask.slice(remainingSpace).map((m) => (m ? 1 : 0)),
                        0
                    );
                    mask.push(newMask);
                }

                state.offset = tokenArray.length - remainingSpace;
            } else {
                currentTokens.set(tokenArray, state.offset);
                if (currentMask && !Array.isArray(tokens)) {
                    currentMask.set(
                        tokens.mask.map((m) => (m ? 1 : 0)),
                        state.offset
                    );
                }
                state.offset += tokenArray.length;
            }
        }
    }
}

export async function tokensFromTasks(
    tasks: Task[],
    tokenizer: ITokeniser,
    cb?: (tokens: number) => void
): Promise<Uint16Array[]>;
export async function tokensFromTasks(
    tasks: Task[],
    tokenizer: ITokeniser,
    cb?: (tokens: number) => void,
    masking?: boolean
): Promise<{ tokens: Uint16Array[]; mask: Uint8Array[] }>;
export async function tokensFromTasks(
    tasks: Task[],
    tokenizer: ITokeniser,
    cb?: (tokens: number) => void,
    masking?: boolean
): Promise<Uint16Array[] | { tokens: Uint16Array[]; mask: Uint8Array[] }> {
    const SHARD_SIZE = 10_000 * 1024; // 10 million tokens

    const allTokens = [new Uint16Array(SHARD_SIZE)];
    const mask: Uint8Array[] | null = masking ? [new Uint8Array(SHARD_SIZE)] : null;
    const state = {
        offset: 0,
        total: 0,
    };

    let lastYield = performance.now();
    while (true) {
        await roundRobinData(tasks, allTokens, tokenizer, state, SHARD_SIZE, mask || undefined);
        // Break if all tasks are exhausted
        if (tasks.every((task) => !task.hasMoreConversations())) {
            break;
        }
        // Yield if more than 40ms has passed
        lastYield = await yieldIfNeeded(lastYield, cb, state.total);
    }

    if (allTokens.length === 1) {
        if (mask) {
            return { tokens: [allTokens[0].subarray(0, state.offset)], mask: [mask[0].subarray(0, state.offset)] };
        }
        return [allTokens[0].subarray(0, state.offset)];
    } else {
        // Truncate the last array to the actual size
        allTokens[allTokens.length - 1] = allTokens[allTokens.length - 1].subarray(0, state.offset);
        if (mask) {
            mask[mask.length - 1] = mask[mask.length - 1].subarray(0, state.offset);
            return { tokens: allTokens, mask };
        }
        return allTokens;
    }
}
