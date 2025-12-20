import TeachableLLM from '@base/TeachableLLM';
import { concat, expandDims, mul, sum, Tensor2D, tensor2d, Tensor3D } from '@tensorflow/tfjs-core';

const BATCH_SIZE = 16;

export function meanPooling(embeddings: Tensor3D, attentionMask?: Tensor2D): Tensor2D {
    if (!attentionMask) {
        // Simple mean pooling over sequence dimension
        const pooled = embeddings.mean(1) as Tensor2D;
        return pooled;
    }

    // Masked mean pooling
    // Expand mask to match embedding dimensions: [batchSize, seqLen] -> [batchSize, seqLen, 1]
    const expandedMask = expandDims(attentionMask, 2);

    // Multiply embeddings by mask to zero out padding
    const maskedEmbeddings = mul(embeddings, expandedMask);

    // Sum over sequence dimension
    const sumEmbeddings = sum(maskedEmbeddings, 1);

    // Count non-padding tokens per sequence
    const tokenCounts = sum(attentionMask, 1, true); // Keep dims for broadcasting

    // Divide by counts to get mean (with small epsilon to avoid division by zero)
    const pooledEmbeddings = sumEmbeddings.div(tokenCounts.maximum(1e-9)) as Tensor2D;
    expandedMask.dispose();
    maskedEmbeddings.dispose();
    sumEmbeddings.dispose();
    tokenCounts.dispose();
    return pooledEmbeddings;
}

export async function sentenceEmbeddingsTensor(
    model: TeachableLLM,
    sentences: string[],
    batchSize: number = BATCH_SIZE
): Promise<Tensor2D> {
    // Generate tokens and batches from sentence strings
    const tokeniser = model.tokeniser;
    const contextLength = model.config.blockSize;

    let resultTensor: Tensor2D | null = null;

    let currentIndex = 0;
    while (currentIndex < sentences.length) {
        const batchSentences = sentences.slice(currentIndex, currentIndex + BATCH_SIZE);
        const sentenceTokens = (await tokeniser.tokenise(batchSentences, true)) as number[][];

        // Truncate or pad tokens to context length and track attention mask
        const inputTokens: number[][] = [];
        const attentionMask: number[][] = [];

        for (const tokens of sentenceTokens) {
            if (tokens.length > contextLength) {
                inputTokens.push(tokens.slice(tokens.length - contextLength, tokens.length));
                attentionMask.push(new Array(contextLength).fill(1));
            } else if (tokens.length < contextLength) {
                inputTokens.push(tokens.concat(new Array(contextLength - tokens.length).fill(0)));
                attentionMask.push(
                    new Array(tokens.length).fill(1).concat(new Array(contextLength - tokens.length).fill(0))
                );
            } else {
                inputTokens.push(tokens);
                attentionMask.push(new Array(contextLength).fill(1));
            }
        }

        // Create input tensor
        const inputTensor = tensor2d(inputTokens, [inputTokens.length, contextLength], 'int32');
        const maskTensor = tensor2d(attentionMask, [attentionMask.length, contextLength], 'float32');

        // Run forward pass
        const embeddings = model.model.forward({ skipLogits: true, training: false }, inputTensor)[0] as Tensor3D;

        const pooledEmbeddings = meanPooling(embeddings, maskTensor);

        // Cleanup intermediate tensors
        maskTensor.dispose();
        embeddings.dispose();

        if (resultTensor === null) {
            resultTensor = pooledEmbeddings;
        } else {
            const oldResult = resultTensor;
            resultTensor = concat([resultTensor, pooledEmbeddings], 0) as Tensor2D;
            oldResult.dispose();
            pooledEmbeddings.dispose();
        }

        currentIndex += batchSize;
    }

    return resultTensor!;
}

export async function sentenceEmbeddings(
    model: TeachableLLM,
    sentences: string[],
    batchSize: number = BATCH_SIZE
): Promise<number[][]> {
    const embeddingsTensor = await sentenceEmbeddingsTensor(model, sentences, batchSize);
    const embeddingsArray = await embeddingsTensor.array();
    embeddingsTensor.dispose();
    return embeddingsArray;
}
