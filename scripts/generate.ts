import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import fs from 'fs';
import path from 'path';
// Note: This should come first due to reimporting issues with TensorFlow
import * as tf from '@tensorflow/tfjs-node-gpu';
import TeachableLLM from '../lib/TeachableLLM';
import chalk from 'chalk';
import waitForModel from '../lib/utilities/waitForModel';

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
        default: 1,
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
    const modelBlob = model.startsWith('http') ? model : fs.readFileSync(path.resolve(model));
    const nanoGPT = TeachableLLM.loadModel(tf, modelBlob);

    await waitForModel(nanoGPT);

    process.stdout.write('\n\n');
    process.stdout.write(chalk.bold(prompt));

    const generator = nanoGPT.generator();

    generator.on('tokens', (tokens: number[], text: string) => {
        process.stdout.write(chalk.yellowBright(text));
    });
    await generator.generate(prompt, {
        maxLength: 500,
        temperature,
    });

    process.stdout.write('\n\n');
}

generate().catch((error) => {
    console.error('Error:', error);
    process.exit(1);
});
