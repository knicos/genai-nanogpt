import type { ITokeniser } from '../tokeniser/type';
import type TF from '@tensorflow/tfjs';

// Training data utilities using TensorFlow.js Dataset API
export class DatasetBuilder {
    public tokenizer: ITokeniser;
    public blockSize: number;
    private tf: typeof TF;

    constructor(tf: typeof TF, tokenizer: ITokeniser, blockSize: number = 128) {
        this.tokenizer = tokenizer;
        this.blockSize = blockSize;
        this.tf = tf;
    }

    // Create dataset from text files
    public async createTextDataset(
        textData: string[],
        batchSize: number = 32,
        start: number = 0,
        end: number = 1
    ): Promise<TF.data.Dataset<{ xs: TF.Tensor; ys: TF.Tensor }>> {
        // Process ALL text into one token array first
        const tokenisedTexts = await Promise.all(textData.map((text) => this.tokenizer.encode(text)));
        // Flatten and add EOS token
        const hasEOS = this.tokenizer.eosToken >= 0;
        const allTokens = tokenisedTexts
            .map((t) => (hasEOS ? [...t, this.tokenizer.eosToken] : t))
            .flat()
            .slice(
                Math.floor(start * tokenisedTexts.length),
                end === 1 ? undefined : Math.floor(end * tokenisedTexts.length)
            );

        // Use generator to avoid storing all sequences in memory
        const generator = function* (this: DatasetBuilder) {
            while (true) {
                const i = Math.floor(Math.random() * (allTokens.length - this.blockSize - 1));
                const xs = allTokens.slice(i, i + this.blockSize);
                const ys = allTokens.slice(i + 1, i + this.blockSize + 1);
                yield { xs, ys };
            }
        }.bind(this);

        return this.tf.data
            .generator(generator)
            .batch(batchSize)
            .map((batch) => {
                // Only needed to convert from float32 to int32
                const batchData = batch as { xs: TF.Tensor; ys: TF.Tensor };
                return this.tf.tidy(() => ({
                    xs: batchData.xs.cast('int32'),
                    ys: this.tf.oneHot(batchData.ys.cast('int32'), this.tokenizer.vocabSize),
                }));
            })
            .prefetch(2); // Smaller prefetch to reduce memory pressure
    }
}
