import { Conversation, ITokeniser } from '@base/main';

export abstract class Task {
    abstract get length(): number;
    abstract hasMoreConversations(): boolean;
    abstract nextConversation(): Conversation[] | null;
    abstract estimateTokens(tokeniser: ITokeniser): Promise<number>;
}

function roundRobinData(
    tasks: Task[],
    allTokens: Uint16Array[],
    tokenizer: ITokeniser,
    state: { offset: number },
    estimatedTokens: number
) {
    // Step through each task in round-robin fashion
    for (let t = 0; t < tasks.length; t++) {
        const convs = tasks[t].nextConversation();
        if (convs) {
            const tokens = tokenizer.encodeConversation(convs);

            const currentTokens = allTokens[allTokens.length - 1];

            // Fill existing array and/or create a new array for remaining tokens if needed
            if (state.offset + tokens.length > currentTokens.length) {
                const remainingSpace = currentTokens.length - state.offset;
                currentTokens.set(tokens.slice(0, remainingSpace), state.offset);
                const newArray = new Uint16Array(Math.floor(estimatedTokens * 0.1) + 100);
                newArray.set(tokens.slice(remainingSpace), 0);
                allTokens.push(newArray);
                state.offset = tokens.length - remainingSpace;
            } else {
                currentTokens.set(tokens, state.offset);
                state.offset += tokens.length;
            }
        }
    }
}

export async function tokensFromTasks(tasks: Task[], tokenizer: ITokeniser): Promise<Uint16Array> {
    const estimatedTokens = (await Promise.all(tasks.map((task) => task.estimateTokens(tokenizer)))).reduce(
        (sum, val) => sum + val,
        0
    );

    const allTokens = [new Uint16Array(estimatedTokens)];
    const state = {
        offset: 0,
    };

    let lastYield = performance.now();
    while (state.offset < estimatedTokens) {
        roundRobinData(tasks, allTokens, tokenizer, state, estimatedTokens);
        // Break if all tasks are exhausted
        if (tasks.every((task) => !task.hasMoreConversations())) {
            break;
        }
        // Yield if more than 40ms has passed
        const now = performance.now();
        if (now - lastYield > 40) {
            await new Promise(requestAnimationFrame);
            lastYield = performance.now();
        }
    }

    if (allTokens.length === 1) {
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

    return finalTokens;
}
