import type TF from '@tensorflow/tfjs';
import { PositionEmbedding } from '@tensorflow/tfjs-layers/dist/layers/nlp/modeling/position_embedding';
import { defaultConfig, GPTConfig } from './config';

import zip from 'jszip';
import { exportWeights, importWeights, ITensorSpec, IWeightManifest } from './weights';
import { ITokeniser } from './Tokeniser/type';
import NodeTokeniser from './Tokeniser/NodeTokeniser';
import Block from './TransformerBlock';

function dummyPass(model: NanoGPT) {
    // Send a dummy input to initialize the model
    const tf = model.tf;
    const dummyInput = tf.zeros([1, model.config.blockSize], 'int32');
    const { logits, loss } = model.forward(dummyInput, undefined, false); // Initialize weights
    logits.dispose(); // Dispose logits to free memory
    if (loss) {
        loss.dispose(); // Dispose loss if it was computed
    }
    dummyInput.dispose(); // Dispose dummy input to free memory
}

export interface TrainingLogEntry {
    epoch: number;
    loss: number;
    valLoss?: number;
    step: number;
    time: number;
    example?: string;
    batchSize: number;
}

// Main GPT model
export default class NanoGPT {
    public readonly config: GPTConfig;
    private wte: TF.layers.Layer; // Token embeddings
    private wpe: PositionEmbedding;
    private drop: TF.layers.Layer; // Dropout
    private blocks: Block[];
    private lnF: TF.layers.Layer; // Final layer norm
    private lmHead: TF.layers.Layer; // Language model head
    public readonly tf: typeof TF;
    public readonly tokeniser: ITokeniser;
    public log: TrainingLogEntry[] = []; // Training log

    constructor(tf: typeof TF, tokeniser: ITokeniser, config: Partial<GPTConfig> = {}) {
        this.tf = tf;
        this.config = { ...defaultConfig, ...config };
        this.tokeniser = tokeniser;

        // Token embeddings
        this.wte = this.tf.layers.embedding({
            inputDim: this.config.vocabSize,
            outputDim: this.config.nEmbed,
            name: 'token_embedding',
        });

        // Position embeddings
        this.wpe = new PositionEmbedding({
            sequenceLength: this.config.blockSize,
        });

        this.drop = this.tf.layers.dropout({ rate: this.config.dropout });

        // Transformer blocks
        this.blocks = [];
        for (let i = 0; i < this.config.nLayer; i++) {
            this.blocks.push(new Block(this.tf, i, this.config));
        }

        // Final layer norm
        this.lnF = this.tf.layers.layerNormalization({
            axis: -1,
            epsilon: 1e-5,
            center: this.config.biasInLayerNorm,
            scale: true,
            name: 'final_layer_norm',
        });

        // Language model head
        this.lmHead = this.tf.layers.dense({
            units: this.config.vocabSize,
            useBias: false,
            name: 'final_output',
        });

        //this.depth = 1; // Initialize depth to 1
    }

    private saveWeights(): Map<string, TF.Tensor[]> {
        const map = new Map<string, TF.Tensor[]>();
        map.set('token_embedding', this.wte.getWeights());
        map.set('positional_embedding', this.wpe.getWeights());
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].saveWeights(map);
        }
        map.set('final_layer_norm', this.lnF.getWeights());
        map.set('final_output', this.lmHead.getWeights());
        return map;
    }

    private loadWeights(weights: Map<string, TF.Tensor[]>): void {
        this.wte.setWeights(weights.get('token_embedding') || []);
        this.wpe.setWeights(weights.get('positional_embedding') || []);
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].loadWeights(weights);
        }
        this.lnF.setWeights(weights.get('final_layer_norm') || []);
        this.lmHead.setWeights(weights.get('final_output') || []);
    }

    async saveModel(): Promise<Blob> {
        const weights = this.saveWeights();
        const zipFile = new zip();

        const spec: Record<string, ITensorSpec[]> = {};

        for (const [name, tensorList] of weights) {
            const data = await exportWeights(tensorList);
            spec[name] = data.spec;
            zipFile.file(`${name}.bin`, data.data.buffer, { binary: true });
        }
        zipFile.file('manifest.json', JSON.stringify({ weightSpec: spec, config: this.config }), {
            binary: false,
        });
        zipFile.file(
            'tokeniser.json',
            JSON.stringify({ vocab: await this.tokeniser.getVocab(), merges: await this.tokeniser.getMerges() }),
            {
                binary: false,
            }
        );
        zipFile.file('log.json', JSON.stringify(this.log), { binary: false });
        return zipFile.generateAsync({ type: 'blob' });
    }

    static async loadModel(tf: typeof TF, blob: Blob | Buffer): Promise<NanoGPT> {
        const zipFile = await zip.loadAsync(blob);
        const manifests = new Map<string, IWeightManifest>();

        const manifestFile = await zipFile.file('manifest.json')?.async('string');
        if (!manifestFile) {
            throw new Error('Manifest file not found in the zip archive');
        }
        const manifest = JSON.parse(manifestFile) as {
            weightSpec: Record<string, ITensorSpec[]>;
            config: GPTConfig;
            vocab: string[];
        };
        for (const [name, specs] of Object.entries(manifest.weightSpec)) {
            manifests.set(name, { spec: specs, data: new Float32Array() });
        }

        const tokeniserFile = await zipFile.file('tokeniser.json')?.async('string');
        if (!tokeniserFile) {
            throw new Error('Tokeniser file not found in the zip archive');
        }
        const tokeniserData = JSON.parse(tokeniserFile) as {
            vocab: string[];
            merges: [string, string][];
        };

        const tokeniser = new NodeTokeniser(tokeniserData.vocab, tokeniserData.merges);

        const weights = new Map<string, TF.Tensor[]>();

        for (const fileName of Object.keys(zipFile.files)) {
            if (fileName.endsWith('.bin')) {
                const name = fileName.replace('.bin', '');
                const data = await zipFile.file(fileName)!.async('arraybuffer');
                const floatData = new Float32Array(data);
                const entry = manifests.get(name) || { spec: [], data: new Float32Array() };
                entry.data = floatData;
                manifests.set(name, entry);

                const tensors = await importWeights(entry, tf);
                weights.set(name, tensors);
            }
        }

        const model = new NanoGPT(tf, tokeniser, manifest.config);

        dummyPass(model); // Initialize the model to set up weights and caches
        model.loadWeights(weights);
        dummyPass(model); // Run a dummy pass to ensure everything is initialized correctly

        const logFile = await zipFile.file('log.json')?.async('string');
        if (logFile) {
            try {
                const logData: TrainingLogEntry[] = JSON.parse(logFile);
                model.log = logData;
            } catch (error) {
                console.error('Error parsing training log:', error);
                throw new Error(`Failed to parse training log: ${error}`);
            }
        }

        return model;
    }

    private inputPhase(idx: TF.Tensor, training: boolean = false): TF.Tensor {
        return this.tf.tidy(() => {
            // Token and position embeddings
            const tokEmb = this.wte.apply(idx) as TF.Tensor; // (b, t, n_embd)
            const posEmb = this.wpe.apply(tokEmb) as TF.Tensor; // (b, t, n_embd)

            // Add embeddings
            const embSum = tokEmb.add(posEmb);

            // Apply dropout
            return this.drop.apply(embSum, { training }) as TF.Tensor;
        });
    }

    setSkipMask(mask: boolean[]): void {
        if (mask.length !== this.blocks.length) {
            throw new Error(`Mask length ${mask.length} does not match number of blocks ${this.blocks.length}`);
        }
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].skipped = mask[i];
        }
    }

    setTrainableMask(mask: boolean[]): void {
        if (mask.length !== this.blocks.length) {
            throw new Error(`Mask length ${mask.length} does not match number of blocks ${this.blocks.length}`);
        }
        for (let i = 0; i < this.blocks.length; i++) {
            this.blocks[i].trainable = mask[i];
        }
    }

    // Forward pass
    forward(idx: TF.Tensor, targets?: TF.Tensor, training: boolean = false): { logits: TF.Tensor; loss?: TF.Tensor } {
        if (idx.shape.length !== 2) {
            throw new Error(`Invalid input shape: expected [batch_size, sequence_length], got ${idx.shape}`);
        }
        if (idx.shape[1] > this.config.blockSize) {
            throw new Error(`Input sequence length ${idx.shape[1]} isn't block size ${this.config.blockSize}`);
        }
        return this.tf.tidy(() => {
            const [, t] = idx.shape;

            if (t > this.config.blockSize) {
                throw new Error(`Cannot forward sequence of length ${t}, block size is only ${this.config.blockSize}`);
            }

            // Input phase
            let x = this.inputPhase(idx, training);

            // const blockTensors: TF.Tensor[] = [x];

            // Apply transformer blocks
            /*for (let i = 0; i < this.blocks.length; i++) {
                const input = blockTensors[i];
                const out = this.blocks[i].call(input);
                blockTensors.push(out);
                x = out;
            }
            for (let i = blockTensors.length - 1; i > 1; i--) {
                const input = blockTensors[i];
                x = this.blocks[i - 2].call(input);
            }*/

            for (const block of this.blocks) {
                x = block.call(x) as TF.Tensor;
            }

            // Final layer norm
            x = this.lnF.apply(x) as TF.Tensor;

            // Language model head
            const logits = this.lmHead.apply(x) as TF.Tensor;

            let loss: TF.Tensor | undefined;
            if (targets) {
                try {
                    const batchSize = logits.shape[0];
                    const seqLength = logits.shape[1]!;

                    // Validate input shapes
                    if (
                        targets.shape.length !== 3 ||
                        targets.shape[0] !== batchSize ||
                        targets.shape[1] !== seqLength
                    ) {
                        throw new Error(
                            `Invalid target shape: expected [${batchSize}, ${seqLength}], got [${targets.shape.join(
                                ', '
                            )}]`
                        );
                    }

                    loss = this.tf.losses.softmaxCrossEntropy(targets, logits);
                } catch (error) {
                    console.error('Error computing loss:', error);
                    throw new Error(`Loss computation failed: ${error}`);
                }
            }

            return { logits, loss };
        });
    }

    // Generate next token
    generate(idx: TF.Tensor, temperature: number = 1.0, topK?: number): TF.Tensor {
        return this.tf.tidy(() => {
            const currentIdx = idx;

            // Crop sequence if it exceeds block size
            const seqLen = currentIdx.shape[1]!;
            const cropIdx =
                seqLen <= this.config.blockSize
                    ? currentIdx
                    : currentIdx.slice(
                          [0, seqLen - this.config.blockSize],
                          [currentIdx.shape[0], this.config.blockSize]
                      );

            // Get predictions
            const { logits } = this.forward(cropIdx, undefined, false);

            // Focus only on the last time step
            const lastTimeStep = logits.shape[1]! - 1;
            const lastLogits = logits.slice([0, lastTimeStep, 0], [logits.shape[0], 1, logits.shape[2]!]); // (b, 1, vocab_size)

            // Apply temperature
            const scaledLogits = lastLogits.div(temperature);

            // Optionally apply top-k filtering
            let nextToken: TF.Tensor;
            if (topK) {
                const { values: topKValues, indices: topKIndices } = this.tf.topk(scaledLogits, topK);
                const topKProbs = this.tf.softmax(topKValues);

                // Sample from top-k
                const sampledIdx = this.tf.multinomial(topKProbs.squeeze([1]) as TF.Tensor1D, 1);
                nextToken = this.tf.gather(topKIndices.squeeze([1]), sampledIdx, 1);
            } else {
                // Sample from full distribution
                const probs = this.tf.softmax(scaledLogits);
                nextToken = this.tf.multinomial(probs.squeeze([1]) as TF.Tensor1D, 1);
            }

            // Ensure nextToken has the right shape (batch_size, 1)
            nextToken = nextToken.reshape([1, 1]);

            return nextToken;
        });
    }

    // Get number of parameters
    getNumParams(): number {
        // This is a simplified count - in practice you'd iterate through all layers
        const embeddingParams = this.config.vocabSize * this.config.nEmbed + this.config.blockSize * this.config.nEmbed;
        const attentionParams =
            this.config.nLayer *
            (4 * this.config.nEmbed * this.config.nEmbed + // qkv + proj
                2 * this.config.nEmbed); // layer norms
        const mlpParams =
            this.config.nLayer *
            (4 * this.config.nEmbed * this.config.nEmbed + // fc
                this.config.nEmbed * 4 * this.config.nEmbed); // proj
        const finalParams = this.config.nEmbed + this.config.vocabSize * this.config.nEmbed;

        return embeddingParams + attentionParams + mlpParams + finalParams;
    }
}
