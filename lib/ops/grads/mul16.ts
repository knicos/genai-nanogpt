import { broadcast_util, GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';
import { mul16 } from '../mul16';
import { sum16 } from '../sum16';
import { reshape16 } from '../reshape16';

const mulGradConfig: GradConfig = {
    kernelName: 'Mul16',
    inputsToSave: ['a', 'b'],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        const [a, b] = saved;
        const outShape = broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

        if (Array.isArray(dy)) {
            throw new Error(`Mul16 gradFunc expected dy to be a Tensor but got an array`);
        }

        const derA = () => {
            let res = mul16(dy, b); // dL/da = dy * b
            const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
            if (reduceAxes.length > 0) {
                const reduced = sum16(res, reduceAxes);
                res.dispose();
                res = reduced;
            }
            const result = reshape16(res, a.shape);
            res.dispose();
            return result;
        };

        const derB = () => {
            let res = mul16(dy, a); // dL/db = dy * a
            const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
            if (reduceAxes.length > 0) {
                const reduced = sum16(res, reduceAxes);
                res.dispose();
                res = reduced;
            }
            const result = reshape16(res, b.shape);
            res.dispose();
            return result;
        };

        return { a: derA, b: derB };
    },
};

registerGradient(mulGradConfig);
