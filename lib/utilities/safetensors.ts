import { DataTypeMap, tensor, Tensor } from '@tensorflow/tfjs-core';

type DType = 'F32' | 'I32';

interface SafeTensorEntry {
    dtype: DType;
    shape: number[];
    data_offsets: [number, number];
}

interface BaseSafeTensorStruct {
    __metadata__?: Record<string, string>;
}

interface SafeTensorEntries {
    [key: string]: SafeTensorEntry;
}

type SafeTensorStruct = BaseSafeTensorStruct & SafeTensorEntries;

function mapTFJSDataType(dtype: keyof DataTypeMap): DType {
    if (dtype === 'float32') return 'F32';
    if (dtype === 'int32') return 'I32';
    throw new Error(`Unsupported dtype: ${dtype}`);
}

function reverseMapDType(dtype: DType): keyof DataTypeMap {
    if (dtype === 'F32') return 'float32';
    if (dtype === 'I32') return 'int32';
    throw new Error(`Unsupported dtype: ${dtype}`);
}

export async function save_safetensors(tensors: Record<string, Tensor>): Promise<ArrayBuffer> {
    const structure: SafeTensorStruct = {};

    let offset = 0;
    for (const [name, t] of Object.entries(tensors)) {
        structure[name] = {
            dtype: mapTFJSDataType(t.dtype),
            shape: t.shape,
            data_offsets: [offset, offset + t.size * 4],
        };
        offset += t.size * 4; // Assuming 4 bytes per element for simplicity
    }

    const jsonHeader = JSON.stringify(structure);
    let headerBytes = new TextEncoder().encode(jsonHeader);

    // If header length is not multiple of 4, pad it with spaces
    if (headerBytes.length % 4 !== 0) {
        const padding = 4 - (headerBytes.length % 4);
        const paddedHeader = new Uint8Array(headerBytes.length + padding);
        paddedHeader.set(headerBytes);
        for (let i = headerBytes.length; i < paddedHeader.length; i++) {
            paddedHeader[i] = 32; // ASCII space
        }
        headerBytes = paddedHeader;
    }

    const headerLength = headerBytes.length;

    const totalLength = 8 + headerLength + offset;
    const buffer = new ArrayBuffer(totalLength);
    const view = new DataView(buffer);

    // Write header length
    view.setUint32(0, headerLength, true);
    // Write header
    new Uint8Array(buffer, 8, headerLength).set(headerBytes);

    let dataOffset = 8 + headerLength;
    for (const t of Object.values(tensors)) {
        if (t.size === 0) continue; // Skip empty tensors
        const tensorData = await t.data();
        if (t.dtype === 'float32') {
            new Float32Array(buffer, dataOffset, t.size).set(tensorData as Float32Array);
            dataOffset += t.size * 4; // Assuming float32
        } else if (t.dtype === 'int32') {
            new Int32Array(buffer, dataOffset, t.size).set(tensorData as Int32Array);
            dataOffset += t.size * 4; // Assuming int32
        } else {
            throw new Error(`Unsupported dtype: ${t.dtype}`);
        }
    }

    return buffer;
}

export async function load_safetensors(buffer: ArrayBuffer): Promise<Record<string, Tensor>> {
    const view = new DataView(buffer);
    const headerLength = view.getUint32(0, true);
    const headerBytes = new Uint8Array(buffer, 8, headerLength);
    const structure: SafeTensorStruct = JSON.parse(new TextDecoder().decode(headerBytes));

    const tensors: Record<string, Tensor> = {};
    for (const [name, meta] of Object.entries(structure)) {
        if (meta.data_offsets[0] === meta.data_offsets[1]) {
            // Handle empty tensor
            tensors[name] = tensor([], meta.shape, reverseMapDType(meta.dtype));
            continue;
        }
        if (meta.dtype === 'F32') {
            const t = tensor(
                new Float32Array(
                    buffer,
                    meta.data_offsets[0] + 8 + headerLength,
                    (meta.data_offsets[1] - meta.data_offsets[0]) / 4
                ),
                meta.shape,
                reverseMapDType(meta.dtype)
            );
            tensors[name] = t;
        } else if (meta.dtype === 'I32') {
            const t = tensor(
                new Int32Array(
                    buffer,
                    meta.data_offsets[0] + 8 + headerLength,
                    (meta.data_offsets[1] - meta.data_offsets[0]) / 4
                ),
                meta.shape,
                reverseMapDType(meta.dtype)
            );
            tensors[name] = t;
        } else {
            throw new Error(`Unsupported dtype: ${meta.dtype}`);
        }
    }
    return tensors;
}
