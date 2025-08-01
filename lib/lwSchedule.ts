import { AdamConfig } from './Trainer';

export interface LWSchedule {
    adam: AdamConfig;
    skip: boolean[];
    trainable: boolean[];
}

export const schedule: LWSchedule[][] = [
    [
        {
            adam: {
                learningRateFactor: 1,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [false],
            trainable: [true],
        },
    ],
    [
        {
            adam: {
                learningRateFactor: 1,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [true, false],
            trainable: [false, true],
        },
        {
            adam: {
                learningRateFactor: 1,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [false, false],
            trainable: [true, false],
        },
        {
            adam: {
                learningRateFactor: 1 / 3,
                beta1: 0.95,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [false, false],
            trainable: [true, true],
        },
    ],
    [],
    [
        {
            adam: {
                learningRateFactor: 1,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [true, true, true, false],
            trainable: [false, false, false, true],
        },
        {
            adam: {
                learningRateFactor: 1,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [true, true, false, false],
            trainable: [false, false, true, false],
        },
        {
            adam: {
                learningRateFactor: 1 / 3,
                beta1: 0.95,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [true, true, false, false],
            trainable: [false, false, false, true],
        },
        {
            adam: {
                learningRateFactor: 1,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [true, false, false, false],
            trainable: [false, true, false, false],
        },
        {
            adam: {
                learningRateFactor: 1 / 3,
                beta1: 0.95,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [true, false, false, false],
            trainable: [false, false, true, false],
        },
        {
            adam: {
                learningRateFactor: 1 / 6,
                beta1: 0.98,
                beta2: 0.9999,
                epsilon: 1e-8,
            },
            skip: [true, false, false, false],
            trainable: [false, false, false, true],
        },
        {
            adam: {
                learningRateFactor: 1,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [false, false, false, false],
            trainable: [true, false, false, false],
        },
        {
            adam: {
                learningRateFactor: 1 / 3,
                beta1: 0.95,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            skip: [false, false, false, false],
            trainable: [false, true, false, false],
        },
        {
            adam: {
                learningRateFactor: 1 / 6,
                beta1: 0.98,
                beta2: 0.9999,
                epsilon: 1e-8,
            },
            skip: [false, false, false, false],
            trainable: [false, false, true, false],
        },
        {
            adam: {
                learningRateFactor: 1 / 6,
                beta1: 0.98,
                beta2: 0.9999,
                epsilon: 1e-8,
            },
            skip: [false, false, false, false],
            trainable: [false, false, false, true],
        },
        {
            adam: {
                learningRateFactor: 1 / 6,
                beta1: 0.98,
                beta2: 0.9999,
                epsilon: 1e-8,
            },
            skip: [false, false, false, false],
            trainable: [true, true, true, true],
        },
    ],
];
