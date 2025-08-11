import type TF from '@tensorflow/tfjs';
import NanoGPT from './NanoGPTModel';
import { ITokeniser } from './tokeniser/type';
import EE from 'eventemitter3';

export interface IGenerateOptions {
    maxLength?: number; /// Maximum length of the generated text
    temperature?: number; /// Controls randomness in generation (default: 1.0)
    topK?: number; /// Limits the number of tokens to consider at each step (default: undefined)
    usePadding?: boolean; /// Whether to use padding in the input tensor (default: false)
    includeAttention?: boolean; /// Whether to include attention in the output (default: false)
    includeProbabilities?: boolean; /// Whether to include probabilities in the output (default: false)
}

const TOKEN_BLOCK_COUNT = 4; // Number of tokens to generate loop

export default class Generator extends EE<'start' | 'stop' | 'tokens'> {
    constructor(private readonly model: NanoGPT, private readonly tokeniser: ITokeniser) {
        super();
    }

    private generateBlockOfTokens(
        inputTensor: TF.Tensor,
        options?: IGenerateOptions
    ): { output: TF.Tensor; attention?: TF.Tensor; probabilities?: TF.Tensor } {
        const temperature = options?.temperature ?? 1.0;
        const topK = options?.topK;
        const usePadding = options?.usePadding ?? options?.includeAttention ?? false;
        const includeAttention = options?.includeAttention ?? false;
        const includeProbabilities = options?.includeProbabilities ?? false;

        let tensor = inputTensor;
        let attention: TF.Tensor | undefined;
        let probabilities: TF.Tensor | undefined;

        // Generate text
        for (let i = 0; i < TOKEN_BLOCK_COUNT; i++) {
            const {
                output: generatedTokens,
                attention: newAttention,
                probabilities: newProbabilities,
            } = this.model.generate(tensor, {
                temperature,
                topK,
                usePadding,
                includeAttention,
                includeProbabilities,
            });
            const oldInput = tensor;
            tensor = this.model.tf.concat([tensor, generatedTokens], 1);

            // Accumulate attention if they are included
            if (attention && newAttention) {
                const oldAttention = attention;
                attention = this.model.tf.concat([attention, newAttention], 0);
                oldAttention.dispose();
            } else if (newAttention) {
                attention = newAttention;
            }

            // Accumulate probabilities if they are included
            if (probabilities && newProbabilities) {
                const oldProbabilities = probabilities;
                probabilities = this.model.tf.concat([probabilities, newProbabilities], 0);
                oldProbabilities.dispose();
            } else if (newProbabilities) {
                probabilities = newProbabilities;
            }

            oldInput.dispose();
            generatedTokens.dispose();
        }

        return { output: tensor, attention, probabilities };
    }

    public async generate(prompt?: string, options?: IGenerateOptions): Promise<string> {
        const tokenisedPrompt = prompt ? await this.tokeniser.tokenise([prompt], true) : [[this.tokeniser.eosToken]];

        let inputTensor: TF.Tensor = this.model.tf.tensor2d(tokenisedPrompt, [1, tokenisedPrompt[0].length], 'int32');

        this.emit('start');

        let outputText = prompt || '';

        // Loop in the model to generate text until eos or max length
        while (true) {
            const { output, attention, probabilities } = this.generateBlockOfTokens(inputTensor, options);
            const oldInput = inputTensor;
            inputTensor = output;

            const newTokens = output.slice([0, oldInput.shape[1]!], [1, TOKEN_BLOCK_COUNT]);
            const newTokensArray = ((await newTokens.array()) as number[][])[0];

            oldInput.dispose();
            newTokens.dispose();

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

            let attentionArray: number[][] | undefined;
            if (attention) {
                attentionArray = (await attention.array()) as number[][];
                attention.dispose();
                if (attentionArray.length > newTokensArray.length) {
                    attentionArray = attentionArray.slice(0, newTokensArray.length);
                }
            }

            let probabilitiesArray: number[][] | undefined;
            if (probabilities) {
                probabilitiesArray = (await probabilities.array()) as number[][];
                probabilities.dispose();
                if (probabilitiesArray.length > newTokensArray.length) {
                    probabilitiesArray = probabilitiesArray.slice(0, newTokensArray.length);
                }
            }

            this.emit('tokens', newTokensArray, newText, attentionArray, probabilitiesArray);

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
