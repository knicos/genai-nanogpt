// Note: This should come first due to reimporting issues with TensorFlow
import * as tf from '@tensorflow/tfjs-node-gpu';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import loadTextData from '../lib/utilities/textLoader';
import fs from 'fs';
import path from 'path';
import { TrainingLogEntry } from '../lib/NanoGPTModel';
import chalk from 'chalk';
import dayjs from 'dayjs';
import duration from 'dayjs/plugin/duration';
import TeachableLLM from '../lib/TeachableLLM';

dayjs.extend(duration);

//tf.enableDebugMode();

const argv = yargs(hideBin(process.argv))
    .option('epochs', {
        alias: 'e',
        type: 'number',
        description: 'Number of training epochs',
        default: 5,
    })
    .option('batch', {
        alias: 'b',
        type: 'number',
        description: 'Batch size for training',
        default: 32,
    })
    .option('rate', {
        alias: 'r',
        type: 'number',
        description: 'Learning rate for the optimizer',
        default: 1e-3,
    })
    .option('maxSteps', {
        type: 'number',
        description: 'Maximum training steps',
        default: 1000000,
    })
    .options('loss', {
        alias: 'l',
        type: 'number',
        description: 'Desired loss to stop training',
        default: 0.01,
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
    .option('blockSteps', {
        type: 'number',
        description: 'Steps per block before moving to next block',
        default: 800,
    })
    .option('vocabSize', {
        type: 'number',
        description: 'Number of tokens to use',
        default: 512,
    })
    .option('context', {
        alias: 'c',
        type: 'number',
        description: 'Context window size in tokens',
        default: 32,
    })
    .option('layers', {
        type: 'number',
        description: 'Number of transformer layers',
        default: 4,
    })
    .option('autosave', {
        type: 'number',
        description: 'Save model at an interval',
        default: 200,
    })
    .parseSync();

async function constructModel(modelPath?: string): Promise<TeachableLLM> {
    if (modelPath) {
        const modelBlob = fs.readFileSync(path.resolve(modelPath));
        console.log('Loading model from:', modelPath);
        const model = await TeachableLLM.loadModel(tf, modelBlob);
        return model;
    }

    const model = TeachableLLM.create(tf, {
        vocabSize: 200,
        blockSize: 128, // Context window size
        nLayer: 4,
        nHead: 3,
        nEmbed: 192,
        dropout: 0.0,
    });
    return model;
}

async function train() {
    const { epochs, batch, data, maxSteps, loss, autosave, rate, model: modelName } = argv;

    if (data === '') {
        console.error('Error: --data option is required');
        process.exit(1);
        return;
    }

    const rawdata = fs.readFileSync(path.resolve(data), 'utf8');
    const textData = await loadTextData(rawdata);

    const model = await constructModel(modelName);
    const tokeniser = model.tokeniser;
    await tokeniser.train(textData);

    const trainer = model.trainer();

    trainer.on('log', async (log: TrainingLogEntry) => {
        console.log(
            `${chalk.bold('Time')} ${dayjs.duration(log.time).asMinutes().toFixed(0)} minutes: ${chalk.bold(
                'Step'
            )} ${chalk.blueBright(log.step)}, ${chalk.bold('Loss:')} ${chalk.redBright(
                log.loss.toFixed(4)
            )}, ${chalk.bold('Example:')}\n${chalk.yellowBright(log.example || 'N/A')}`
        );

        if (log.step > 0 && log.step % autosave === 0) {
            try {
                const blob = await model.saveModel();
                fs.writeFileSync('nanogpt_model.zip', Buffer.from(await blob.arrayBuffer()));
                console.log('\nModel Saved\n');
            } catch (error) {
                console.error('Autosave failed', error);
            }
        }
    });

    await trainer.train(textData, {
        epochs,
        prompt: 'What a great movie. It',
        maxSteps,
        logInterval: 10,
        validationSplit: 0.2,
        desiredLoss: loss,
        batchSize: batch,
        learningRate: rate,
    });

    // Save the model after training
    const modelBlob = await model.saveModel();
    fs.writeFileSync('nanogpt_model.zip', Buffer.from(await modelBlob.arrayBuffer()));
    console.log('Training completed!');
}

train().catch((error) => {
    console.error('Error:', error);
    process.exit(1);
});
