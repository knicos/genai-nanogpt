import NanoGPT from '@base/NanoGPTModel';
import loadTransformers, { TransformersConfig, TransformersMetadata, TransformersTokeniser } from './loadTransformers';
import { ITokeniser } from '@base/main';

export default async function loadHuggingFace(
    name: string
): Promise<{ model: NanoGPT; tokeniser: ITokeniser; name?: string }> {
    const configUrl = `https://huggingface.co/${name}/resolve/main/config.json`;
    const tokenUrl = `https://huggingface.co/${name}/resolve/main/tokeniser.json`;
    const metaUrl = `https://huggingface.co/${name}/resolve/main/meta.json`;
    const weightsUrl = `https://huggingface.co/${name}/resolve/main/model.safetensors`;

    const [configResponse, tokenResponse, metaResponse, weightsResponse] = await Promise.all([
        fetch(configUrl),
        fetch(tokenUrl),
        fetch(metaUrl),
        fetch(weightsUrl),
    ]);

    if (!configResponse.ok) {
        throw new Error(`Failed to fetch config from ${configUrl}: ${configResponse.statusText}`);
    }
    if (!tokenResponse.ok) {
        throw new Error(`Failed to fetch tokeniser from ${tokenUrl}: ${tokenResponse.statusText}`);
    }
    if (!metaResponse.ok) {
        throw new Error(`Failed to fetch meta from ${metaUrl}: ${metaResponse.statusText}`);
    }
    if (!weightsResponse.ok) {
        throw new Error(`Failed to fetch weights from ${weightsUrl}: ${weightsResponse.statusText}`);
    }

    const config = (await configResponse.json()) as TransformersConfig;
    const tokeniser = (await tokenResponse.json()) as TransformersTokeniser;
    const meta = (await metaResponse.json()) as TransformersMetadata;
    const weightData = await weightsResponse.arrayBuffer();

    return loadTransformers(config, tokeniser, meta, weightData);
}
