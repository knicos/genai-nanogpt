import NanoGPT from '@base/NanoGPTModel';

export async function dummyPassAsync(model: NanoGPT) {
    // Send a dummy input to initialize the model
    const tf = model.tf;
    const dummyInput = tf.zeros([1, model.config.blockSize], 'int32');
    const { logits, loss } = model.forward(dummyInput, undefined, false);
    await logits.data(); // Just to wait
    logits.dispose();
    if (loss) {
        loss.dispose();
    }
    dummyInput.dispose();
}

export function dummyPass(model: NanoGPT) {
    // Send a dummy input to initialize the model
    const tf = model.tf;
    const dummyInput = tf.zeros([1, model.config.blockSize], 'int32');
    const { logits, loss } = model.forward(dummyInput, undefined, false);
    logits.dispose();
    if (loss) {
        loss.dispose();
    }
    dummyInput.dispose();
}
