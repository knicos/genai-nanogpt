import NanoGPT from '@base/NanoGPTModel';
import { engine, memory, Scalar, variableGrads, zeros } from '@tensorflow/tfjs-core';
import { ExtendedMemoryInfo } from './profile';

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

export interface MemoryRequirements {
    perBatch: number;
    tapeSize: number;
    gradients: number;
}

export async function dummyPassTrainAsync(model: NanoGPT): Promise<MemoryRequirements> {
    const startMemInfo = memory() as ExtendedMemoryInfo;
    const startBytes =
        startMemInfo.numBytesInGPUAllocated ?? startMemInfo.numBytesAllocatedInGPU ?? startMemInfo.numBytes;

    await dummyPassAsync(model);
    // Send a dummy input to initialize the model
    const dummyInput = zeros([1, model.config.gpt.blockSize], 'int32');
    const dummyTarget = zeros([1, model.config.gpt.blockSize], 'int32');

    const memoryReqs: MemoryRequirements = {
        perBatch: 0,
        tapeSize: 0,
        gradients: model.getNumParams() * 4,
    };

    const f = () => {
        const [logits, loss] = model.forward({ training: true }, dummyInput, dummyTarget);

        const tape = engine().state.activeTape;
        let totalTapeSize = 0;
        if (tape) {
            for (const item of tape) {
                totalTapeSize += item.saved?.reduce((a, b) => a + b.size * 4, 0) || 0;
            }
        }

        memoryReqs.tapeSize = totalTapeSize;

        logits.dispose();
        return loss! as Scalar;
    };

    //const vars = this.model.variables;
    const { value: lossValue, grads } = variableGrads(f);

    const endMemInfo = memory() as ExtendedMemoryInfo;
    const endBytes = endMemInfo.numBytesInGPUAllocated ?? endMemInfo.numBytesAllocatedInGPU ?? endMemInfo.numBytes;
    memoryReqs.perBatch = endBytes - startBytes - memoryReqs.gradients;

    console.log('Dummy training memory requirements:', memoryReqs);

    await lossValue.data(); // Just to wait
    lossValue.dispose();
    for (const key in grads) {
        grads[key].dispose();
    }
    dummyInput.dispose();
    dummyTarget.dispose();

    return memoryReqs;
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
