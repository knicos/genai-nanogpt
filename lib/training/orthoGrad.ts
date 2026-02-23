import { Tensor, tidy } from '@tensorflow/tfjs-core';

/* Grokking at the Edge of Numerical Stability, Prieto et al. 2025 */
export function orthogonalizeGradient(weight: Tensor, gradient: Tensor, epsilon: number): Tensor {
    return tidy(() => {
        const w = weight.reshape([-1]);
        const g = gradient.reshape([-1]);

        const wNormSq = w.mul(w).sum().add(epsilon);
        const proj = w.mul(g).sum().div(wNormSq);
        const gOrth = g.sub(w.mul(proj));

        const gNorm = g.norm();
        const gOrthNorm = gOrth.norm().add(epsilon);
        const gOrthScaled = gOrth.mul(gNorm.div(gOrthNorm));

        return gOrthScaled.reshape(gradient.shape);
    });
}
