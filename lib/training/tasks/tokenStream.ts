import { Conversation, ITokeniser } from '@base/tokeniser/type';
import { ConversationStream } from '@base/data/stream';
import { createTokenStore, deleteTokenStore, TokenStore } from './TokenStore';
import { seededRng } from '@base/utilities/random';

function tokensFromConversation(
    conv: Conversation[],
    allTokens: Uint16Array[],
    tokenizer: ITokeniser,
    state: { offset: number; total: number },
    estimatedTokens: number,
    mask?: Uint8Array[]
) {
    const tokens = tokenizer.encodeConversation(conv, false, !!mask); //await task.nextTokens(tokenizer, mask ? true : undefined);
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

interface TokensFromTasksOptions {
    masking?: boolean;
    maxCachedShards?: number;
    noOPFS?: boolean;
    shardSize?: number;
    validationSplit?: number;
    validationSeed?: string | number;
    cb?: (tokens: number) => void;
}

export async function tokensFromStreams(
    tasks: ConversationStream[],
    tokenizer: ITokeniser,
    datasetId: string,
    options?: TokensFromTasksOptions
): Promise<{ trainingTokens: TokenStore; validationTokens?: TokenStore }> {
    await deleteTokenStore('training-tokens');
    const trainingStore = await createTokenStore('training-tokens', tokenizer.id, datasetId, options);

    await deleteTokenStore('validation-tokens');
    const validationStore =
        options?.validationSplit && options.validationSplit > 0
            ? await createTokenStore('validation-tokens', tokenizer.id, datasetId, options)
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

    let taskIndex = 0;

    const cb = options?.cb;

    const rng = options?.validationSeed !== undefined ? seededRng(options.validationSeed) : Math.random;

    while (taskIndex < tasks.length) {
        const stream = tasks[taskIndex++];
        //await tokensFromStream(task, allTokens, tokenizer, state, store.shardSize, mask || undefined, options?.cb);

        await stream.begin(
            (conv) => {
                const isValidationPhase =
                    options?.validationSplit && options.validationSplit > 0 && rng() < options.validationSplit;
                const allTokens = isValidationPhase ? validationTokens! : trainingTokens;
                const state = isValidationPhase ? validationState : trainingState;
                const mask = isValidationPhase ? validationMask : trainingMask;
                const store = isValidationPhase ? validationStore! : trainingStore;
                tokensFromConversation(conv, allTokens, tokenizer, state, store.shardSize, mask || undefined);
                if (allTokens.length > 1) {
                    // Append the first shard to the store
                    store.appendShard(allTokens[0], mask ? mask[0] : undefined);
                    // Remove the first shard and reset offset
                    allTokens.shift();
                    if (mask) {
                        mask.shift();
                    }
                }
            },
            cb ? () => cb(trainingState.total) : undefined
        );

        //await roundRobinData(tasks, allTokens, tokenizer, state, store.shardSize, mask || undefined);
    }

    if (trainingTokens.length === 1) {
        // Truncate the first array to the actual size and append to store
        trainingTokens[0] = trainingTokens[0].subarray(0, trainingState.offset);
        trainingStore.appendShard(
            trainingTokens[0],
            trainingMask ? trainingMask[0].subarray(0, trainingState.offset) : undefined
        );
    }
    if (validationTokens && validationTokens.length === 1) {
        // Truncate the first array to the actual size and append to store
        validationTokens[0] = validationTokens[0].subarray(0, validationState.offset);
        validationStore!.appendShard(
            validationTokens[0],
            validationMask ? validationMask[0].subarray(0, validationState.offset) : undefined
        );
    }

    await trainingStore.finish();
    if (validationStore) {
        await validationStore.finish();
    }

    return { trainingTokens: trainingStore, validationTokens: validationTokens ? validationStore : undefined };
}
