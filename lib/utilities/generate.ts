import NanoGPT from '../NanoGPTModel';
import type TF from '@tensorflow/tfjs';

export async function generateText(
    model: NanoGPT,
    prompt: string,
    length: number,
    temperature: number = 1.0,
    topK?: number
): Promise<string> {
    if (length <= 0) {
        throw new Error('Length must be a positive integer');
    }
    if (temperature <= 0) {
        throw new Error('Temperature must be a positive number');
    }
    if (topK !== undefined && topK <= 0) {
        throw new Error('topK must be a positive integer or undefined');
    }
    if (prompt.length === 0) {
        throw new Error('Prompt cannot be an empty string');
    }

    // Tokenise the prompt
    const tokenisedPrompt = await model.tokeniser.tokenise([prompt], true);

    const inputTensor = model.tf.tidy(() => {
        let inputTensor: TF.Tensor = model.tf.tensor2d(tokenisedPrompt, [1, tokenisedPrompt[0].length], 'int32');

        // Generate text
        for (let i = 0; i < length; i++) {
            const generatedTokens = model.generate(inputTensor, temperature, topK);
            const oldInput = inputTensor;
            inputTensor = model.tf.concat([inputTensor, generatedTokens], 1);
            oldInput.dispose();
            generatedTokens.dispose();
        }

        return inputTensor;
    });

    const tokenArray = (await inputTensor.array()) as number[][];

    const generatedTokens = tokenArray[0];

    // Remove anything after the first end-of-sequence token
    const endIndex = generatedTokens.indexOf(model.tokeniser.eosToken);
    if (endIndex !== -1) {
        generatedTokens.splice(endIndex);
    }

    // Decode the generated tokens back to text
    const generatedText = await model.tokeniser.decode(generatedTokens);

    return generatedText;
}
