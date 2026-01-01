import { Tensor } from '@tensorflow/tfjs-core';
import { matMul16 } from './matMul16';
import { transpose16 } from './transpose16';
import { reshape16 } from './reshape16';
import { isPackedTensor } from '@base/utilities/packed';
import { dot } from '@tensorflow/tfjs-layers/dist/backend/tfjs_backend';

export function dot16(a: Tensor, b: Tensor, transposeA = false, transposeB = false): Tensor {
    if (!isPackedTensor(a) && !isPackedTensor(b)) {
        return dot(a, b);
    }

    if (a.rank < 2 || b.rank < 2) {
        throw new Error(
            `dot requires both inputs to be rank >= 2` + ` but got x shape = ${a.shape} and y shape = ${b.shape}`
        );
    }
    if (b.rank >= 3) {
        const xLastDim = a.shape.slice(-1)[0];
        const ySecondLastDim = b.shape.slice(-2)[0];
        if (xLastDim !== ySecondLastDim) {
            throw new Error(
                `If rank y >= 3, then the second last dim` +
                    ` of y must equal the last dim of x but got x shape = ${a.shape} and ` +
                    ` y shape = ${b.shape}`
            );
        }
    }
    if (!isPackedTensor(a) || !isPackedTensor(b)) {
        throw new Error('dot16 requires both inputs to be packed Tensors.');
    }

    // Handle basic 2D x 2D case.
    if (a.rank === 2 && b.rank === 2) {
        // tfc.fused.matMul only fuses certain activation functions. Unsupported
        // activation functions are treated as 'linear' activations, which is
        // equivalent to a no-op.
        return matMul16(a, b, transposeA, transposeB);
    } else {
        // Reshape x into the analogous 2D Tensor.
        const aFirstDims = a.shape.slice(); // Holds all but the last dim of x.
        const aLastDim = aFirstDims.pop()!;
        a = reshape16(a, [-1, aLastDim]);

        // Reshape y into the analogous 2D Tensor, and keep track of the
        // required dimensions to reproduce the output shape.
        const bShape = b.shape.slice();
        const bLastDim = bShape.pop()!;
        const ySecondLastDim = bShape.pop()!;
        const yOtherDims = [...bShape, bLastDim];
        // permutation should be like [r-2, 0, 1, 2, ... r-4, r-3, r-1]
        // where r is the rank of y.
        const perm = Array.from({ length: b.rank }, (_, i) => {
            if (i === 0) {
                return b.rank - 2;
            } else if (i <= b.rank - 2) {
                return i - 1;
            }
            return i;
        });

        const nopTransposeB = perm.every((p, i) => p === i);
        if (nopTransposeB) {
            b = reshape16(b, [ySecondLastDim, -1]);
        } else {
            const bt = transpose16(b, perm);
            b = reshape16(bt, [ySecondLastDim, -1]);
            bt.dispose();
        }

        // Multiply x and y as 2D Tensors, and then reshape back to original.
        const outputShape = [...aFirstDims, ...yOtherDims];
        const m = matMul16(a, b, transposeA, transposeB);
        a.dispose();
        b.dispose();
        const result = reshape16(m, outputShape);
        m.dispose();
        return result;
    }
}
