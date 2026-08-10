import { Conversation, ITokeniser } from '@base/main';
import { yieldIfNeeded } from '@base/utilities/yielder';
import { createTokenStore, deleteTokenStore, TokenStore } from './TokenStore';

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

interface TokensFromTasksOptions {
    masking?: boolean;
    maxCachedShards?: number;
    noOPFS?: boolean;
    shardSize?: number;
    validationSplit?: number;
    cb?: (tokens: number) => void;
}

export async function tokensFromTasks(
    tasks: Task[],
    tokenizer: ITokeniser,
    options?: TokensFromTasksOptions
): Promise<{ trainingTokens: TokenStore; validationTokens?: TokenStore }> {
    await deleteTokenStore('training-tokens');
    const trainingStore = await createTokenStore('training-tokens', tokenizer.id, tokenizer.datasetID ?? '', options);

    await deleteTokenStore('validation-tokens');
    const validationStore =
        options?.validationSplit && options.validationSplit > 0
            ? await createTokenStore('validation-tokens', tokenizer.id, tokenizer.datasetID ?? '', options)
            : undefined;

    const trainingTokens = [new Uint16Array(trainingStore.shardSize)];
    const trainingMask: Uint8Array[] | null = options?.masking ? [new Uint8Array(trainingStore.shardSize)] : null;
    const trainingState = {
        offset: 0,
        total: 0,
    };

    const validationTokens =
        options?.validationSplit && options.validationSplit > 0
            ? [new Uint16Array(validationStore!.shardSize)]
            : undefined;
    const validationMask: Uint8Array[] | null =
        options?.masking && validationTokens ? [new Uint8Array(validationStore!.shardSize)] : null;
    const validationState = {
        offset: 0,
        total: 0,
    };

    let lastYield = performance.now();
    while (true) {
        const isValidationPhase =
            options?.validationSplit && options.validationSplit > 0 && Math.random() < options.validationSplit;
        const allTokens = isValidationPhase ? validationTokens! : trainingTokens;
        const state = isValidationPhase ? validationState : trainingState;
        const mask = isValidationPhase ? validationMask : trainingMask;
        const store = isValidationPhase ? validationStore! : trainingStore;
        await roundRobinData(tasks, allTokens, tokenizer, state, store.shardSize, mask || undefined);

        if (allTokens.length > 1) {
            // Append the first shard to the store
            store.appendShard(allTokens[0], mask ? mask[0] : undefined);
            // Remove the first shard and reset offset
            allTokens.shift();
            if (mask) {
                mask.shift();
            }
        }

        // Break if all tasks are exhausted
        if (tasks.every((task) => !task.hasMoreConversations())) {
            break;
        }

        // Yield if more than 40ms has passed
        lastYield = await yieldIfNeeded(lastYield, options?.cb, trainingState.total);
    }

    if (trainingTokens.length === 1) {
        // Truncate the first array to the actual size and append to store
        trainingTokens[0] = trainingTokens[0].subarray(0, trainingState.offset);
        await trainingStore.appendShard(
            trainingTokens[0],
            trainingMask ? trainingMask[0].subarray(0, trainingState.offset) : undefined
        );
    }
    if (validationTokens && validationTokens.length === 1) {
        // Truncate the first array to the actual size and append to store
        validationTokens[0] = validationTokens[0].subarray(0, validationState.offset);
        await validationStore!.appendShard(
            validationTokens[0],
            validationMask ? validationMask[0].subarray(0, validationState.offset) : undefined
        );
    }

    return { trainingTokens: trainingStore, validationTokens: validationTokens ? validationStore : undefined };
}
