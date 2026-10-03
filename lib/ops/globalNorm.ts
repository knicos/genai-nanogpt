import { Scalar, Tensor, engine, tidy, zeros, scalar } from '@tensorflow/tfjs-core';

export function clipScale(grads: Tensor[], invLossScaling: number, clipNorm = 1.0): Scalar {
    if (engine().backendName !== 'webgpu') {
        return scalar(invLossScaling);
    }

    return tidy(() => {
        const output = zeros([grads.length], 'int32');
        grads.map((grad, index) =>
            engine().runKernel('Norm2', { x: grad, output }, { invLossScaling, index })
        ) as Tensor[];

        return engine().runKernel('ClipScale', { x: output }, { invLossScaling, clipNorm });
    });
}
