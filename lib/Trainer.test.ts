import { beforeEach, describe, it, vi } from 'vitest';
import Trainer from './Trainer';
import CharTokeniser from './tokeniser/CharTokeniser';
import { createTrainValidationDatasets } from './training/validation';
import { DatasetMetadata, TransformersMetadata } from './loader/types';
import Model, { ModelForwardAttributes } from './models/model';

vi.mock('./training/validation', () => ({
    createTrainValidationDatasets: vi.fn(),
}));

function createMockModel() {
    return {
        config: { blockSize: 8, nLayer: 2 },
        lossScaling: 1,
        trainingState: null,
        metaData: {} as TransformersMetadata,

        hasLoRA: vi.fn(() => true),
        detachLoRA: vi.fn(),
        createLoRA: vi.fn(),
        attachLoRA: vi.fn(),
        deleteLoRA: vi.fn(),

        weightStore: {
            setTrainable: vi.fn(),
            touchVariables: vi.fn(),
        },

        // Not needed for this test path, but useful if something touches profiler
        getProfiler: vi.fn(() => undefined),
        setProfiler: vi.fn(),

        // Forward/project/dispose are only needed if you execute real BasicTrainer math
        forward: vi.fn(),
        project: vi.fn(),
        dispose: vi.fn(),
    };
}

describe('Trainer Tests', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('prepares pre-training data correctly', async ({ expect }) => {
        const tokeniser = new CharTokeniser(64);
        const mockModel = createMockModel();

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (createTrainValidationDatasets as any).mockResolvedValue({
            trainDataset: { iterator: vi.fn() }, // shape only; we stub trainOnDataset below
            validationDataset: { iterator: vi.fn() }, // shape only
            size: 100,
        });

        const trainer = new Trainer(mockModel as unknown as Model<ModelForwardAttributes>, tokeniser, 'pretraining', {
            batchSize: 2,
            learningRate: 1e-3,
            sftMode: 'full',
        });

        // Avoid real tensor training loop; just verify Trainer orchestration
        /*const innerTrainer = trainer['trainer'];
        vi.spyOn(innerTrainer, 'trainOnDataset').mockResolvedValue({
            losses: [],
            validationLosses: [],
        });*/

        const rawData = new Uint16Array([1, 2, 3, 4]);
        const validationData = new Uint16Array([5, 6, 7, 8]);
        await trainer.prepare([rawData], [validationData], [{ id: 'ds1', conversational: false } as DatasetMetadata]);

        expect(createTrainValidationDatasets).toHaveBeenCalledWith(
            [rawData],
            [validationData],
            tokeniser,
            expect.anything(), // datasetBuilder
            2
        );

        expect(mockModel.metaData.pretrainingData).toEqual([{ id: 'ds1', conversational: false }]);
        expect(mockModel.metaData.mode).toBe('completion');
    });

    it('prepares conversational data correctly', async ({ expect }) => {
        const tokeniser = new CharTokeniser(64);
        const mockModel = createMockModel();

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (createTrainValidationDatasets as any).mockResolvedValue({
            trainDataset: { iterator: vi.fn() }, // shape only; we stub trainOnDataset below
            validationDataset: { iterator: vi.fn() }, // shape only
            size: 100,
        });

        const trainer = new Trainer(mockModel as unknown as Model<ModelForwardAttributes>, tokeniser, 'pretraining', {
            batchSize: 2,
            learningRate: 1e-3,
            sftMode: 'full',
        });

        // Avoid real tensor training loop; just verify Trainer orchestration
        /*const innerTrainer = trainer['trainer'];
        vi.spyOn(innerTrainer, 'trainOnDataset').mockResolvedValue({
            losses: [],
            validationLosses: [],
        });*/

        const rawData = new Uint16Array([1, 2, 3, 4]);
        const validationData = new Uint16Array([5, 6, 7, 8]);
        await trainer.prepare([rawData], [validationData], [{ id: 'ds1', conversational: true } as DatasetMetadata]);

        expect(createTrainValidationDatasets).toHaveBeenCalledWith(
            [rawData],
            [validationData],
            tokeniser,
            expect.anything(), // datasetBuilder
            2
        );

        expect(mockModel.metaData.pretrainingData).toEqual([{ id: 'ds1', conversational: true }]);
        expect(mockModel.metaData.mode).toBe('conversational');
    });
});
