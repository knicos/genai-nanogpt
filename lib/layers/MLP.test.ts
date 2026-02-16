import { afterEach, describe, it } from 'vitest';
import MLP from './MLP';
import * as tf from '@tensorflow/tfjs';
import { GPTConfig } from '@base/main';

describe('MLP', () => {
    afterEach(() => {
        tf.disposeVariables();
    });

    it('should call the MLP layer and return a tensor', ({ expect }) => {
        const config: GPTConfig = {
            modelType: 'GenAI_NanoGPT_v2',
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            vocabSize: 20,
            blockSize: 4,
            mlpFactor: 4,
        };
        const mlp = new MLP(0, config, { activation: 'relu2' });

        const input = tf.randomNormal([1, 4, 16]);
        const output = mlp.call({ training: false }, input) as tf.Tensor;

        expect(output).toBeInstanceOf(tf.Tensor);
        expect(output.shape).toEqual([1, 4, 16]);
    });

    it('should save and load weights correctly', ({ expect }) => {
        const config: GPTConfig = {
            modelType: 'GenAI_NanoGPT_v2',
            nEmbed: 16,
            nHead: 2,
            nLayer: 1,
            vocabSize: 20,
            blockSize: 4,
            mlpFactor: 4,
        };
        const mlp = new MLP(0, config, { activation: 'gelu' });
        const input = tf.randomNormal([1, 4, 16]);
        mlp.call({ training: false }, input); // Initialize the layer

        // Touch all variables to simulate training
        mlp.weightStore.touchVariables(mlp.weightStore.variableNames);

        const weightsMap = new Map<string, tf.Tensor[]>();
        mlp.weightStore.saveWeights(weightsMap);
        const originalOutput = mlp.call({ training: false }, input) as tf.Tensor;

        mlp.dispose();

        const newMlp = new MLP(0, config, { activation: 'gelu' });
        newMlp.call({ training: false }, input); // Initialize the layer
        newMlp.weightStore.loadWeights(weightsMap, false);

        const newOutput = newMlp.call({ training: false }, input) as tf.Tensor;
        expect(originalOutput.shape).toEqual(newOutput.shape);
        expect(originalOutput.dataSync()).toEqual(newOutput.dataSync());

        newMlp.dispose();
        input.dispose();
        originalOutput.dispose();
        newOutput.dispose();
    });
});
