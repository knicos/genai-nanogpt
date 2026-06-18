import { Conversation, ITokeniser } from '@base/main';
import { yieldIfNeeded } from '@base/utilities/yielder';

export abstract class Task {
    abstract get length(): number;
    abstract hasMoreConversations(): boolean;
    abstract nextConversation(): Conversation[] | null;

    abstract nextTokens(tokeniser: ITokeniser): number[] | null;
    abstract nextTokens(tokeniser: ITokeniser, masking: boolean): { tokens: number[]; mask: boolean[] } | null;
    abstract nextTokens(
        tokeniser: ITokeniser,
        masking?: boolean
    ): number[] | { tokens: number[]; mask: boolean[] } | null;

    abstract estimateTokens(tokeniser: ITokeniser): Promise<number>;
    abstract shuffle(): void;
}

function roundRobinData(
    tasks: Task[],
    allTokens: Uint16Array[],
    tokenizer: ITokeniser,
    state: { offset: number; total: number },
    estimatedTokens: number,
    mask?: Uint8Array[]
) {
    // Step through each task in round-robin fashion
    for (const task of tasks) {
        const tokens = task.nextTokens(tokenizer, mask ? true : undefined);
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
                const newArray = new Uint16Array(Math.max(Math.floor(estimatedTokens * 0.1) + 100, neededSize));
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
): Promise<Uint16Array>;
export async function tokensFromTasks(
    tasks: Task[],
    tokenizer: ITokeniser,
    cb?: (tokens: number) => void,
    masking?: boolean
): Promise<{ tokens: Uint16Array; mask: Uint8Array }>;
export async function tokensFromTasks(
    tasks: Task[],
    tokenizer: ITokeniser,
    cb?: (tokens: number) => void,
    masking?: boolean
): Promise<Uint16Array | { tokens: Uint16Array; mask: Uint8Array }> {
    const estimatedTokens = Math.min(
        (await Promise.all(tasks.map((task) => task.estimateTokens(tokenizer)))).reduce((sum, val) => sum + val, 0),
        tokenizer.vocabSize * 10000
    );

    const allTokens = [new Uint16Array(estimatedTokens)];
    const mask: Uint8Array[] | null = masking ? [new Uint8Array(estimatedTokens)] : null;
    const state = {
        offset: 0,
        total: 0,
    };

    let lastYield = performance.now();
    while (state.offset < estimatedTokens) {
        roundRobinData(tasks, allTokens, tokenizer, state, estimatedTokens, mask || undefined);
        // Break if all tasks are exhausted
        if (tasks.every((task) => !task.hasMoreConversations())) {
            break;
        }
        // Yield if more than 40ms has passed
        lastYield = await yieldIfNeeded(lastYield, cb, state.total);
    }

    if (allTokens.length === 1) {
        if (mask) {
            return { tokens: allTokens[0].subarray(0, state.offset), mask: mask[0].subarray(0, state.offset) };
        }
        return allTokens[0].subarray(0, state.offset);
    }

    // Combine all arrays into one
    const totalLength =
        allTokens.reduce((sum, arr) => sum + arr.length, 0) - (allTokens[allTokens.length - 1].length - state.offset);
    const finalTokens = new Uint16Array(totalLength);
    let pos = 0;
    for (let i = 0; i < allTokens.length; i++) {
        const arr = allTokens[i];
        if (i === allTokens.length - 1) {
            finalTokens.set(arr.subarray(0, state.offset), pos);
            pos += state.offset;
        } else {
            finalTokens.set(arr, pos);
            pos += arr.length;
        }
    }

    if (mask) {
        const finalMask = new Uint8Array(totalLength);
        pos = 0;
        for (let i = 0; i < mask.length; i++) {
            const arr = mask[i];
            if (i === mask.length - 1) {
                finalMask.set(arr.subarray(0, state.offset), pos);
                pos += state.offset;
            } else {
                finalMask.set(arr, pos);
                pos += arr.length;
            }
        }
        return { tokens: finalTokens, mask: finalMask };
    }

    return finalTokens;
}
