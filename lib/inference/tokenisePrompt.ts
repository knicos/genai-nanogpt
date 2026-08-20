import type { Conversation, ITokeniser } from '../tokeniser/type';
import { IGenerateOptions } from './types';
import { tensor2d, Tensor } from '@tensorflow/tfjs-core';

export default async function tokenisePrompt(
    tokeniser: ITokeniser,
    blockSize: number,
    prompt?: Conversation[],
    options?: IGenerateOptions
): Promise<Tensor> {
    if (prompt) {
        const isAssistant = prompt.length > 0 && prompt[prompt.length - 1].role === 'text';
        let tokenisedPrompt: number[];
        if (options?.nonConversational) {
            if (isAssistant && options?.continuation) {
                tokenisedPrompt = [tokeniser.bosToken, ...tokeniser.encode(prompt[prompt.length - 1].content)];
            } else {
                tokenisedPrompt = tokeniser.encodeAsSequence(prompt, true);
            }
        } else {
            tokenisedPrompt = tokeniser.encodeConversation(prompt, true);
        }
        if (tokenisedPrompt.length > blockSize) {
            tokenisedPrompt = tokenisedPrompt.slice(-blockSize);
        }

        const inputTensor: Tensor = tensor2d([tokenisedPrompt], [1, tokenisedPrompt.length], 'int32');
        return inputTensor;
    } else {
        const startToken = options?.nonConversational
            ? undefined
            : tokeniser.getSpecialTokenIndex('<|assistant_start|>');
        const tokenisedPrompt = startToken !== undefined ? [tokeniser.bosToken, startToken] : [tokeniser.bosToken];
        const inputTensor: Tensor = tensor2d([tokenisedPrompt], [1, tokenisedPrompt.length], 'int32');
        return inputTensor;
    }
}
