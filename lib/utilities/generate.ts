import { ITokeniser } from '@base/tokeniser/type';
import NanoGPT, { GenerateOptions } from '../NanoGPTModel';
import type TF from '@tensorflow/tfjs';
import { KVCache } from '@base/layers/CausalSelfAttention';

export async function generateText(
    tokeniser: ITokeniser,
    model: NanoGPT,
    prompt: string,
    length: number,
    options: GenerateOptions
): Promise<string> {
    if (length <= 0) {
        throw new Error('Length must be a positive integer');
    }
    if (prompt.length === 0) {
        throw new Error('Prompt cannot be an empty string');
    }

    // Tokenise the prompt
    const tokenisedPrompt = await tokeniser.tokenise([prompt], true);

    const cache: KVCache[] | undefined = model.config.useRope
        ? new Array(model.config.nLayer).fill(undefined)
        : undefined;

    const inputTensor = model.tf.tidy(() => {
        let inputTensor: TF.Tensor = model.tf.tensor2d(tokenisedPrompt, [1, tokenisedPrompt[0].length], 'int32');
        let outputTensor = inputTensor;

        // Generate text
        for (let i = 0; i < length; i++) {
            const { output: generatedTokens } = model.generate(inputTensor, cache, options);
            const oldInput = inputTensor;
            const oldOutput = outputTensor;
            outputTensor = model.tf.concat([outputTensor, generatedTokens], 1);

            inputTensor = cache ? generatedTokens : model.tf.concat([inputTensor, generatedTokens], 1);
            oldInput.dispose();
            oldOutput.dispose();
            if (!cache) generatedTokens.dispose();
        }

        return outputTensor;
    });

    const tokenArray = (await inputTensor.array()) as number[][];
    inputTensor.dispose();

    const generatedTokens = tokenArray[0];

    // Remove anything after the first end-of-sequence token
    const endIndex = generatedTokens.indexOf(tokeniser.eosToken);
    if (endIndex !== -1) {
        generatedTokens.splice(endIndex);
    }

    // Decode the generated tokens back to text
    const generatedText = await tokeniser.decode(generatedTokens);

    return generatedText;
}
