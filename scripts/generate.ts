import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import fs from 'fs';
import path from 'path';
// Note: This should come first due to reimporting issues with TensorFlow
import * as tf from '@tensorflow/tfjs-node-gpu';
import NanoGPT from '../lib/NanoGPTModel';
import chalk from 'chalk';

const argv = yargs(hideBin(process.argv))
    .option('model', {
        alias: 'm',
        type: 'string',
        description: 'Path to the trained model file',
        default: '',
    })
    .option('prompt', {
        alias: 'p',
        type: 'string',
        description: 'Prompt text to start generation',
        default: '',
    })
    .option('length', {
        type: 'number',
        description: 'Number of tokens to generate',
        default: 50,
    })
    .option('temperature', {
        alias: 't',
        type: 'number',
        description: 'Temperature for sampling',
        default: 0.8,
    })
    .parseSync();

async function generate() {
    const { model, prompt, length, temperature } = argv;

    if (model === '') {
        console.error('Error: --model option is required for generation');
        return;
    }
    if (prompt === '') {
        console.error('Error: --prompt option is required for generation');
        return;
    }
    if (length <= 0) {
        console.error('Error: --length must be a positive number');
        return;
    }
    if (temperature <= 0) {
        console.error('Error: --temperature must be a positive number');
        return;
    }

    // Load the trained model
    const modelBlob = fs.readFileSync(path.resolve(model));
    const nanoGPT = await NanoGPT.loadModel(tf, modelBlob);

    const tokeniser = nanoGPT.tokeniser;
    process.stdout.write('\n\n');
    process.stdout.write(chalk.bold(prompt));

    // Tokenise the prompt
    const tokenisedPrompt = await tokeniser.tokenise([prompt], true);
    let inputTensor: tf.Tensor = tf.tensor2d(tokenisedPrompt, [1, tokenisedPrompt[0].length]);

    // Generate text
    for (let i = 0; i < length; i++) {
        const generatedTokens = nanoGPT.generate(inputTensor, temperature, 10);
        const tokenArray = generatedTokens.arraySync() as number[][];
        const casted = generatedTokens.cast('float32');
        generatedTokens.dispose();

        const generatedText = await tokeniser.decode(tokenArray[0]);
        if (generatedText === '<eos>') break; // Stop if end of sequence token is generated
        process.stdout.write(chalk.yellowBright(generatedText));

        // Concatenate the new token to the input tensor for next iteration

        const newInputTensor = tf.concat([inputTensor, casted], 1);

        // Dispose of old tensors to prevent memory leaks
        inputTensor.dispose();
        casted.dispose();

        inputTensor = newInputTensor;
    }

    process.stdout.write('\n\n');

    inputTensor.dispose();
}

generate().catch((error) => {
    console.error('Error:', error);
    process.exit(1);
});
