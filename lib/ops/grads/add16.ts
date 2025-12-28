import { broadcast_util, GradConfig, registerGradient, Tensor } from '@tensorflow/tfjs-core';
import { sum16 } from '../sum16';
import { reshape16 } from '../reshape16';

const addGradConfig: GradConfig = {
    kernelName: 'Add16',
    inputsToSave: ['a', 'b'],
    gradFunc: (dy: Tensor | Tensor[], saved: Tensor[]) => {
        const [a, b] = saved;
        const outShape = broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);

        if (Array.isArray(dy)) {
            throw new Error(`Add16 gradFunc expected dy to be a Tensor but got an array`);
        }

        const derA = () => {
            let res = dy;
            const reduceAxes = broadcast_util.getReductionAxes(a.shape, outShape);
            if (reduceAxes.length > 0) {
                res = sum16(res, reduceAxes);
            }
            return reshape16(res, a.shape);
        };
        const derB = () => {
            let res = dy;
            const reduceAxes = broadcast_util.getReductionAxes(b.shape, outShape);
            if (reduceAxes.length > 0) {
                res = sum16(res, reduceAxes);
            }
            return reshape16(res, b.shape);
        };

        return { a: derA, b: derB };
    },
};

registerGradient(addGradConfig);
