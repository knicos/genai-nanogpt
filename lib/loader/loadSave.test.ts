import { describe, it, expect, vi, afterEach } from 'vitest';
import zip from 'jszip';
import { tensor, Tensor } from '@tensorflow/tfjs-core';
import { saveModel } from './save';
import { loadModel, VERSION } from './load';
import { load_safetensors, save_safetensors } from '@base/utilities/safetensors';
import type { ITokeniser } from '@base/tokeniser/type';
import type Model from '@base/models/model';
import type { ModelForwardAttributes } from '@base/models/model';
import loadZipFile from './newZipLoad';
import { TransformersMetadata } from './types';
import '@tensorflow/tfjs';
import createModelInstance from '@base/models/factory';
import { CharTokeniser } from '@base/main';

vi.mock('./newZipLoad', () => ({ default: vi.fn() }));
vi.mock('./oldZipLoad', () => ({ default: vi.fn() }));
vi.mock('./loadHF', () => ({ default: vi.fn() }));

afterEach(() => {
    vi.resetAllMocks();
    vi.unstubAllGlobals();
});

describe('save/load zips', () => {
    it('saveModel writes a reference model', async () => {
        const model = createModelInstance({
            modelType: 'GenAI_NanoGPT_v1',
            vocabSize: 20,
            blockSize: 16,
            nLayer: 1,
            nHead: 1,
            nEmbed: 4,
            dropout: 0,
            biasInLinear: false,
            biasInLayerNorm: false,
            mlpFactor: 4,
            useRope: false,
        });

        model.metaData = {
            version: VERSION,
            application: '@genai-fi/nanogpt',
            url: 'http://example.com/model',
        };

        const tokeniser = new CharTokeniser(20);

        const blob = await saveModel(model, tokeniser, {
            name: 'tiny',
            metadata: { purpose: 'test' },
        });

        const zipFile = await zip.loadAsync(blob);
        expect(zipFile.file('model.safetensors')).toBeTruthy();
        expect(zipFile.file('config.json')).toBeTruthy();
        expect(zipFile.file('meta.json')).toBeTruthy();
        expect(zipFile.file('tokeniser.json')).toBeTruthy();

        const config = JSON.parse(await zipFile.file('config.json')!.async('string'));
        expect(config.model_type).toBe('GenAI_NanoGPT_v1');
        expect(config.vocab_size).toBe(20);
        expect(config.hidden_size).toBe(4);

        const meta = JSON.parse(await zipFile.file('meta.json')!.async('string'));
        expect(meta.version).toBe(VERSION);
        expect(meta.name).toBe('tiny');
        expect(meta.meta).toEqual({ purpose: 'test' });
        expect(meta.reference).toBe('http://example.com/model');

        const tokeniserJson = JSON.parse(await zipFile.file('tokeniser.json')!.async('string'));
        expect(tokeniserJson.vocab).toContain('<|user_start|>');

        // Check that no weights were saved since it's a reference model
        const safetensorsFile = zipFile.file('model.safetensors');
        expect(safetensorsFile).toBeTruthy();
        const safetensorsData = await safetensorsFile!.async('arraybuffer');
        const weights = await load_safetensors(safetensorsData);
        expect(Object.keys(weights).length).toBe(0);

        model.dispose();
    });

    it('saveModel writes only changed weights', async () => {
        const model = createModelInstance({
            modelType: 'GenAI_NanoGPT_v1',
            vocabSize: 20,
            blockSize: 16,
            nLayer: 1,
            nHead: 1,
            nEmbed: 4,
            dropout: 0,
            biasInLinear: false,
            biasInLayerNorm: false,
            mlpFactor: 4,
            useRope: false,
        });

        model.metaData = {
            version: VERSION,
            application: '@genai-fi/nanogpt',
            url: 'http://example.com/model',
        };

        const tokeniser = new CharTokeniser(20);

        // Change one weight to simulate training
        model.weightStore.touchVariables(['block_0_rms1']);

        const blob = await saveModel(model, tokeniser, {
            name: 'tiny',
            metadata: { purpose: 'test' },
        });

        const zipFile = await zip.loadAsync(blob);
        expect(zipFile.file('model.safetensors')).toBeTruthy();
        expect(zipFile.file('config.json')).toBeTruthy();
        expect(zipFile.file('meta.json')).toBeTruthy();
        expect(zipFile.file('tokeniser.json')).toBeTruthy();

        const config = JSON.parse(await zipFile.file('config.json')!.async('string'));
        expect(config.model_type).toBe('GenAI_NanoGPT_v1');
        expect(config.vocab_size).toBe(20);
        expect(config.hidden_size).toBe(4);

        const meta = JSON.parse(await zipFile.file('meta.json')!.async('string'));
        expect(meta.version).toBe(VERSION);
        expect(meta.name).toBe('tiny');
        expect(meta.meta).toEqual({ purpose: 'test' });
        expect(meta.reference).toBe('http://example.com/model');

        const tokeniserJson = JSON.parse(await zipFile.file('tokeniser.json')!.async('string'));
        expect(tokeniserJson.vocab).toContain('<|user_start|>');

        // Check that no weights were saved since it's a reference model
        const safetensorsFile = zipFile.file('model.safetensors');
        expect(safetensorsFile).toBeTruthy();
        const safetensorsData = await safetensorsFile!.async('arraybuffer');
        const weights = await load_safetensors(safetensorsData);
        const keys = Object.keys(weights);
        expect(keys.length).toBe(1);
        expect(keys[0]).toBe('block_0_rms1');

        model.dispose();
    });

    it('loadModel merges weights from a reference zip', async () => {
        const refWeights = { w: tensor([9], [1], 'float32') };
        const refWeightsBin = (await save_safetensors(refWeights)) as ArrayBuffer;

        const referenceZip = new zip();
        referenceZip.file('model.safetensors', refWeightsBin, { binary: true });
        referenceZip.file(
            'meta.json',
            JSON.stringify({ version: VERSION, reference: 'https://example.com/base.zip' }),
            { binary: false }
        );
        const referenceBlob = await referenceZip.generateAsync({ type: 'blob' });

        const baseZip = new zip();
        baseZip.file('meta.json', JSON.stringify({ version: VERSION }), { binary: false });
        const baseArrayBuffer = await baseZip.generateAsync({ type: 'arraybuffer' });

        vi.stubGlobal(
            'fetch',
            vi.fn(async () => ({
                ok: true,
                statusText: 'OK',
                arrayBuffer: async () => baseArrayBuffer,
            }))
        );

        const loadWeightsCalls: { map: Map<string, Tensor[]>; reference: boolean }[] = [];
        const stubModel = {
            weightStore: {
                loadWeights: (map: Map<string, Tensor[]>, reference: boolean) => {
                    loadWeightsCalls.push({ map, reference });
                },
            },
            config: {
                modelType: 'TinyTest',
                nEmbed: 2,
                nLayer: 1,
                nHead: 1,
                blockSize: 4,
                dropout: 0,
                biasInLinear: false,
                biasInLayerNorm: false,
                mlpFactor: 4,
                useRope: false,
            },
            getClassName: () => 'TinyTestModel',
        } as unknown as Model<ModelForwardAttributes>;

        const stubTokeniser = {
            getVocab: () => ['a'],
            getMerges: async () => [],
        } as unknown as ITokeniser;

        vi.mocked(loadZipFile).mockResolvedValue({
            model: stubModel,
            tokeniser: stubTokeniser,
            metaData: {} as TransformersMetadata,
        });

        const result = await loadModel(referenceBlob);

        expect(result.model).toBe(stubModel);
        expect(result.metaData.reference).toBe('https://example.com/base.zip');
        expect(loadWeightsCalls.length).toBe(1);
        expect(loadWeightsCalls[0].reference).toBe(false);

        const loadedTensor = loadWeightsCalls[0].map.get('w')?.[0];
        expect(loadedTensor).toBeTruthy();
        const data = await loadedTensor!.data();
        expect(Array.from(data)).toEqual([9]);
    });
});
