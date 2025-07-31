import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import loadTextData from '../lib/utilities/textLoader';
import fs from 'fs';
import path from 'path';
// Note: This should come first due to reimporting issues with TensorFlow
import * as tf from '@tensorflow/tfjs-node-gpu';
import NanoGPT from '../lib/NanoGPTModel';
import chalk from 'chalk';
import FullTrainer from '../lib/FullTrainer';

const argv = yargs(hideBin(process.argv))
    .option('batch', {
        alias: 'b',
        type: 'number',
        description: 'Batch size for training',
        default: 32,
    })
    .options('data', {
        alias: 'd',
        type: 'string',
        description: 'Path to the data',
        default: '',
    })
    .option('model', {
        alias: 'm',
        type: 'string',
        description: 'Path to the trained model file',
        default: '',
    })
    .parseSync();

async function evaluate() {
    const { model, data, batch } = argv;

    if (model === '') {
        console.error('Error: --model option is required for evaluation');
        process.exit(1);
        return;
    }

    if (data === '') {
        console.error('Error: --data option is required for evaluation');
        process.exit(1);
        return;
    }

    // Load the trained model
    const modelBlob = fs.readFileSync(path.resolve(model));
    const nanoGPT = await NanoGPT.loadModel(tf, modelBlob);
    const tokeniser = nanoGPT.tokeniser;

    console.log(`Layers: ${nanoGPT.config.nLayer}`);
    console.log(`Vocab Size: ${tokeniser.vocabSize}`);
    console.log(`Model Parameters: ${nanoGPT.getNumParams()}`);
    console.log(`Context Length: ${nanoGPT.config.blockSize}`);

    // Load and prepare the validation dataset
    const rawdata = fs.readFileSync(path.resolve(data), 'utf8');
    const textData = await loadTextData(rawdata);
    const trainer = new FullTrainer(tf, nanoGPT, tokeniser);
    const splitIndex = Math.floor(textData.length * (1 - 0.1));
    const dataset = await trainer.createDataset(textData.slice(splitIndex), batch);

    // Evaluate the model
    const validationLoss = await trainer.evaluateOnDataset(dataset);
    console.log(`Validation loss: ${chalk.bold.red(validationLoss.toFixed(4))}`);
}

evaluate().catch((error) => {
    console.error('Error:', error);
    process.exit(1);
});
