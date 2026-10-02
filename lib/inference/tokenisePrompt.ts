import type { Conversation, ITokeniser } from '../tokeniser/type';
import { IGenerateOptions } from './types';
import { tensor2d, Tensor } from '@tensorflow/tfjs-core';

export function tensorFromTokens(tokens: number[]): Tensor {
    return tensor2d([tokens], [1, tokens.length], 'int32');
}

export default async function tokenisePrompt(
    tokeniser: ITokeniser,
    blockSize: number,
    prompt?: Conversation[],
    options?: IGenerateOptions
): Promise<number[]> {
    if (prompt) {
        const isAssistant =
            prompt.length > 0 &&
            (prompt[prompt.length - 1].role === 'text' || prompt[prompt.length - 1].role === 'assistant');
        let tokenisedPrompt: number[];
        if (options?.nonConversational) {
            if (isAssistant && options?.continuation) {
                tokenisedPrompt = [tokeniser.bosToken, ...tokeniser.encode(prompt[prompt.length - 1].content)];
            } else {
                tokenisedPrompt = tokeniser.encodeAsSequence(prompt, true);
            }
        } else {
            if (isAssistant && options?.continuation) {
                tokenisedPrompt = tokeniser.encodeConversation(prompt, true);

                if (
                    tokenisedPrompt[tokenisedPrompt.length - 1] ===
                        tokeniser.getSpecialTokenIndex('<|assistant_start|>') &&
                    tokenisedPrompt[tokenisedPrompt.length - 2] === tokeniser.getSpecialTokenIndex('<|assistant_end|>')
                ) {
                    tokenisedPrompt.pop();
                    tokenisedPrompt.pop();
                }
            } else {
                tokenisedPrompt = tokeniser.encodeConversation(prompt, true);
            }
        }
        if (tokenisedPrompt.length > blockSize) {
            tokenisedPrompt = tokenisedPrompt.slice(-blockSize);
        }

        if (options?.debug) {
            console.log('Tokenised prompt:', tokeniser.decode(tokenisedPrompt));
        }

        return tokenisedPrompt;
    } else {
        const startToken = options?.nonConversational
            ? undefined
            : tokeniser.getSpecialTokenIndex('<|assistant_start|>');
        const tokenisedPrompt = startToken !== undefined ? [tokeniser.bosToken, startToken] : [tokeniser.bosToken];
        return tokenisedPrompt;
    }
}
