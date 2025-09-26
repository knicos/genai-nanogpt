import NanoGPT from '@base/NanoGPTModel';
import { zeros } from '@tensorflow/tfjs-core';

export async function dummyPassAsync(model: NanoGPT) {
    // Send a dummy input to initialize the model
    const dummyInput = zeros([1, model.config.gpt.blockSize], 'int32');
    const [logits, loss] = model.forward({ training: false }, dummyInput);
    await logits.data(); // Just to wait
    logits.dispose();
    if (loss) {
        loss.dispose();
    }
    dummyInput.dispose();
}

export function dummyPass(model: NanoGPT) {
    // Send a dummy input to initialize the model
    const dummyInput = zeros([1, model.config.gpt.blockSize], 'int32');
    const [logits, loss] = model.forward({ training: false }, dummyInput);
    logits.dispose();
    if (loss) {
        loss.dispose();
    }
    dummyInput.dispose();
}
