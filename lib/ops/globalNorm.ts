import { Scalar, Tensor, engine, tidy, zeros } from '@tensorflow/tfjs-core';

export function clipScale(grads: Tensor[], invLossScaling: number, clipNorm = 1.0): Scalar {
    return tidy(() => {
        const output = zeros([grads.length], 'int32');
        grads.map((grad, index) =>
            engine().runKernel('Norm2', { x: grad, output }, { invLossScaling, index })
        ) as Tensor[];
        //const summed = output.cast('float32').sum();
        //const globalNorm = summed.div(scalar(100.0, 'float32')).sqrt(); // Dequantize
        /*keep(globalNorm);
        globalNorm.data().then((data) => {
            console.log('Global norm:', data[0]);
            globalNorm.dispose();
        });*/

        return engine().runKernel('ClipScale', { x: output }, { invLossScaling, clipNorm });
        /*const clipNormScalar = scalar(clipNorm);
        const scale = clipNormScalar.div(maximum(globalNorm, clipNormScalar));
        return scale.mul(scalar(invLossScaling));*/
    });
}
