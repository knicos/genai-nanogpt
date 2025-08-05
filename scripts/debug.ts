import * as tf from '@tensorflow/tfjs-node-gpu';
import NanoGPT from '../lib/NanoGPTModel';
import NodeTokeniser from '../lib/Tokeniser/NodeTokeniser';
import fs from 'fs';
import path from 'path';
import Block from '../lib/TransformerBlock';
import MLP from '../lib/MLP';
import CausalSelfAttention from '../lib/CausalSelfAttention';
import AlternateLayerNorm from '../lib/AlternateLayerNorm';
import { TiedEmbeddingOutputLayer } from '../lib/TiedEmbedding';

async function constructModel(modelPath?: string) {
    if (modelPath) {
        const modelBlob = fs.readFileSync(path.resolve(modelPath));
        console.log('Loading model from:', modelPath);
        return NanoGPT.loadModel(tf, modelBlob);
    }
    const tokeniser = new NodeTokeniser();

    const model = new NanoGPT(tf, tokeniser, {
        vocabSize: 256,
        blockSize: 128, // Context window size
        nLayer: 1,
        nHead: 2,
        nEmbed: 128,
        dropout: 0.0,
    });
    return model;
}

function constructBlock(): Block {
    return new Block(tf, 0, {
        nEmbed: 4,
        nHead: 2,
        nLayer: 1,
        vocabSize: 4,
        blockSize: 8,
        dropout: 0,
        biasInLayerNorm: false,
        biasInLinear: false,
    });
}

function constructMLP(): MLP {
    return new MLP(tf, 0, {
        nEmbed: 4,
        nHead: 2,
        nLayer: 1,
        vocabSize: 4,
        blockSize: 8,
        dropout: 0,
        biasInLayerNorm: false,
        biasInLinear: false,
    });
}

function constructAttention(): CausalSelfAttention {
    return new CausalSelfAttention(tf, 0, {
        nEmbed: 4,
        nHead: 2,
        nLayer: 1,
        vocabSize: 4,
        blockSize: 8,
        dropout: 0,
        biasInLayerNorm: false,
        biasInLinear: false,
    });
}

function constructLayerNorm(): AlternateLayerNorm {
    return new AlternateLayerNorm(tf, [4], 1e-5);
}

function constructTokenEmbedding(): TiedEmbeddingOutputLayer {
    return new TiedEmbeddingOutputLayer(tf, {
        embedDim: 4,
        vocabSize: 4,
    });
}

async function debugBlock() {
    const block = constructBlock();

    const seqLen = 8; // Sequence length
    const batchSize = 2; // Batch size
    const nEmbed = 4; // Must match the nEmbed in constructBlock

    // Create input tensors with the same pattern as debugModel
    const singleSample = tf.randomNormal([1, seqLen, nEmbed]);
    const batchedIdentical = tf.tile(singleSample, [batchSize, 1, 1]);

    // Forward pass through the block
    const outputSingle = block.call(singleSample);
    const outputBatched = block.call(batchedIdentical);

    // Extract first element from batched output to compare
    const firstBatchElement = outputBatched.slice([0, 0, 0], [1, seqLen, nEmbed]);
    const diff = tf.mean(tf.abs(tf.sub(outputSingle, firstBatchElement)));

    console.log('Block output difference:', diff.dataSync()[0]);

    // Clean up tensors
    singleSample.dispose();
    batchedIdentical.dispose();
    outputSingle.dispose();
    outputBatched.dispose();
    firstBatchElement.dispose();
    diff.dispose();
}

async function debugMLP() {
    const mlp = constructMLP();

    const seqLen = 8; // Sequence length
    const batchSize = 2; // Batch size
    const nEmbed = 4; // Must match the nEmbed in constructBlock

    // Create input tensors with the same pattern as debugModel
    const singleSample = tf.randomNormal([1, seqLen, nEmbed]);
    const batchedIdentical = tf.tile(singleSample, [batchSize, 1, 1]);

    // Forward pass through the block
    const outputSingle = mlp.call(singleSample);
    const outputBatched = mlp.call(batchedIdentical);

    // Extract first element from batched output to compare
    const firstBatchElement = outputBatched.slice([0, 0, 0], [1, seqLen, nEmbed]);
    const diff = tf.mean(tf.abs(tf.sub(outputSingle, firstBatchElement)));

    console.log('MLP output difference:', diff.dataSync()[0]);

    // Clean up tensors
    singleSample.dispose();
    batchedIdentical.dispose();
    outputSingle.dispose();
    outputBatched.dispose();
    firstBatchElement.dispose();
    diff.dispose();
}

async function debugAttention() {
    const attention = constructAttention();

    const seqLen = 8; // Sequence length
    const batchSize = 2; // Batch size
    const nEmbed = 4; // Must match the nEmbed in constructBlock

    // Create input tensors with the same pattern as debugModel
    const singleSample = tf.randomNormal([1, seqLen, nEmbed]);
    const batchedIdentical = tf.tile(singleSample, [batchSize, 1, 1]);

    // Forward pass through the block
    const outputSingle = attention.call(singleSample);
    const outputBatched = attention.call(batchedIdentical);

    // Extract first element from batched output to compare
    const firstBatchElement = outputBatched.slice([0, 0, 0], [1, seqLen, nEmbed]);
    const diff = tf.mean(tf.abs(tf.sub(outputSingle, firstBatchElement)));

    console.log('Attention output difference:', diff.dataSync()[0]);

    // Clean up tensors
    singleSample.dispose();
    batchedIdentical.dispose();
    outputSingle.dispose();
    outputBatched.dispose();
    firstBatchElement.dispose();
    diff.dispose();
}

async function debugLayerNorm() {
    const layerNorm = constructLayerNorm();

    const seqLen = 8; // Sequence length
    const batchSize = 2; // Batch size
    const nEmbed = 4; // Must match the nEmbed in constructBlock

    // Create input tensors with the same pattern as debugModel
    const singleSample = tf.randomNormal([1, seqLen, nEmbed]);
    const batchedIdentical = tf.tile(singleSample, [batchSize, 1, 1]);

    // Forward pass through the block
    const outputSingle = layerNorm.apply(singleSample);
    const outputBatched = layerNorm.apply(batchedIdentical);

    // Extract first element from batched output to compare
    const firstBatchElement = outputBatched.slice([0, 0, 0], [1, seqLen, nEmbed]);
    const diff = tf.mean(tf.abs(tf.sub(outputSingle, firstBatchElement)));

    console.log('LN output difference:', diff.dataSync()[0]);

    // Clean up tensors
    singleSample.dispose();
    batchedIdentical.dispose();
    outputSingle.dispose();
    outputBatched.dispose();
    firstBatchElement.dispose();
    diff.dispose();
}

async function debugTiedEmbedding() {
    const tiedEmbedding = constructTokenEmbedding();

    const seqLen = 8; // Sequence length
    const batchSize = 2; // Batch size

    const singleSample = tf.tensor2d([0, 1, 2, 3, 0, 1, 2, 3], [1, seqLen], 'int32');
    const batchedIdentical = tf.tile(singleSample, [batchSize, 1]);

    // Forward pass through the block
    const outputSingle = tiedEmbedding.embed(singleSample);
    const outputBatched = tiedEmbedding.embed(batchedIdentical);

    // Extract first element from batched output to compare
    const firstBatchElement = outputBatched.slice([0, 0, 0], [1, seqLen, 4]);
    const diff = tf.mean(tf.abs(tf.sub(outputSingle, firstBatchElement)));

    console.log('Embedding output difference:', diff.dataSync()[0]);

    // Clean up tensors
    singleSample.dispose();
    batchedIdentical.dispose();
    outputSingle.dispose();
    outputBatched.dispose();
    firstBatchElement.dispose();
    diff.dispose();
}

async function debugPosition() {
    const model = await constructModel();

    const seqLen = 8; // Sequence length
    const batchSize = 2; // Batch size

    const singleSample = tf.tensor2d([0, 1, 2, 3, 0, 1, 2, 3], [1, seqLen], 'int32');
    const batchedIdentical = tf.tile(singleSample, [batchSize, 1]);

    // Forward pass through the block
    const outputSingle = model.inputPhase(singleSample) as tf.Tensor;
    const outputBatched = model.inputPhase(batchedIdentical) as tf.Tensor;

    console.log('Shapes', outputSingle.shape, outputBatched.shape);

    // Extract first element from batched output to compare
    const firstBatchElement = outputBatched.slice([0, 0, 0], [1, seqLen, 64]);
    const diff = tf.mean(tf.abs(tf.sub(outputSingle, firstBatchElement)));

    console.log('Input output difference:', diff.dataSync()[0]);

    // Clean up tensors
    singleSample.dispose();
    batchedIdentical.dispose();
    outputSingle.dispose();
    outputBatched.dispose();
    firstBatchElement.dispose();
    diff.dispose();
}

async function debugTiedHead() {
    const tiedEmbedding = constructTokenEmbedding();

    const seqLen = 8; // Sequence length
    const batchSize = 2; // Batch size
    const nEmbed = 4; // Must match the nEmbed in constructBlock

    // Create input tensors with the same pattern as debugModel
    const singleSample = tf.randomNormal([1, seqLen, nEmbed]);
    const batchedIdentical = tf.tile(singleSample, [batchSize, 1, 1]);

    // Forward pass through the block
    const outputSingle = tiedEmbedding.project(singleSample);
    const outputBatched = tiedEmbedding.project(batchedIdentical);

    // Extract first element from batched output to compare
    const firstBatchElement = outputBatched.slice([0, 0, 0], [1, seqLen, 4]);
    const diff = tf.mean(tf.abs(tf.sub(outputSingle, firstBatchElement)));

    console.log('Head output difference:', diff.dataSync()[0]);

    // Clean up tensors
    singleSample.dispose();
    batchedIdentical.dispose();
    outputSingle.dispose();
    outputBatched.dispose();
    firstBatchElement.dispose();
    diff.dispose();
}

async function debugModel() {
    const model = await constructModel();

    const seqLen = 8; // Sequence length
    const batchSize = 2; // Batch size

    const singleSample = tf.tensor2d([0, 1, 2, 3, 0, 1, 2, 3], [1, seqLen], 'int32');
    const batchedIdentical = tf.tile(singleSample, [batchSize, 1]);
    const { logits: logitsSingle } = model.forward(singleSample, undefined, false);
    const { logits: logitsBatched } = model.forward(batchedIdentical, undefined, false);

    const firstBatchElement = logitsBatched.slice([0, 0], [1, seqLen]);
    const diff = tf.mean(tf.abs(tf.sub(logitsSingle, firstBatchElement)));
    console.log('Model Output difference:', diff.dataSync()[0]);
}

await debugModel().catch((error) => {
    console.error('Error during model debug:', error);
});

tf.disposeVariables();

await debugPosition().catch((error) => {
    console.error('Error during attention debug:', error);
});

tf.disposeVariables();

await debugMLP().catch((error) => {
    console.error('Error during attention debug:', error);
});

tf.disposeVariables();

await debugAttention().catch((error) => {
    console.error('Error during attention debug:', error);
});

tf.disposeVariables();

await debugBlock().catch((error) => {
    console.error('Error during attention debug:', error);
});

tf.disposeVariables();

await debugLayerNorm().catch((error) => {
    console.error('Error during attention debug:', error);
});

tf.disposeVariables();

await debugTiedEmbedding().catch((error) => {
    console.error('Error during attention debug:', error);
});

tf.disposeVariables();

await debugTiedHead().catch((error) => {
    console.error('Error during attention debug:', error);
});

tf.disposeVariables();
