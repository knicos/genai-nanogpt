import type TF from '@tensorflow/tfjs';
import NanoGPT from './NanoGPTModel';
import { ITokeniser } from './tokeniser/type';
import EE from 'eventemitter3';

export interface IGenerateOptions {
    maxLength?: number; /// Maximum length of the generated text
    temperature?: number; /// Controls randomness in generation (default: 1.0)
}

const TOKEN_BLOCK_COUNT = 4; // Number of tokens to generate loop

export default class Generator extends EE<'start' | 'stop' | 'tokens'> {
    constructor(private readonly model: NanoGPT, private readonly tokeniser: ITokeniser) {
        super();
    }

    private generateBlockOfTokens(inputTensor: TF.Tensor, options?: IGenerateOptions): TF.Tensor {
        const temperature = options?.temperature ?? 1.0;

        let tensor = inputTensor;
        // Generate text
        for (let i = 0; i < TOKEN_BLOCK_COUNT; i++) {
            const generatedTokens = this.model.generate(tensor, temperature, undefined, true);
            const oldInput = tensor;
            tensor = this.model.tf.concat([tensor, generatedTokens], 1);
            oldInput.dispose();
            generatedTokens.dispose();
        }

        return tensor;
    }

    public async generate(prompt?: string, options?: IGenerateOptions): Promise<string> {
        const tokenisedPrompt = prompt ? await this.tokeniser.tokenise([prompt], true) : [[this.tokeniser.eosToken]];

        let inputTensor: TF.Tensor = this.model.tf.tensor2d(tokenisedPrompt, [1, tokenisedPrompt[0].length], 'int32');

        this.emit('start');

        let outputText = prompt || '';

        // Loop in the model to generate text until eos or max length
        while (true) {
            const output = this.generateBlockOfTokens(inputTensor, options);
            const oldInput = inputTensor;
            inputTensor = output;

            const newTokens = output.slice([0, oldInput.shape[1]!], [1, TOKEN_BLOCK_COUNT]);
            const newTokensArray = ((await newTokens.array()) as number[][])[0];

            let hasEOSToken = false;
            let hasEnough = false;

            // Remove anything after the first end-of-sequence token
            const endIndex = newTokensArray.indexOf(this.tokeniser.eosToken);
            if (endIndex !== -1) {
                hasEOSToken = true;
                newTokensArray.splice(endIndex);
            }
            if (newTokensArray.length + outputText.length >= (options?.maxLength ?? 1000)) {
                hasEnough = true;
                newTokensArray.splice(
                    options?.maxLength ? options.maxLength - outputText.length : newTokensArray.length
                );
            }

            const newText = await this.tokeniser.decode(newTokensArray);
            outputText += newText;
            this.emit('tokens', newTokensArray, newText);

            oldInput.dispose();
            newTokens.dispose();

            if (hasEOSToken) {
                break;
            }
            if (hasEnough) {
                break;
            }
        }

        inputTensor.dispose();
        this.emit('stop');
        return outputText;
    }
}
