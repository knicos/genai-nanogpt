# GenAI NanoGPT

A browser-native implementation of small transformer language models using TensorFlow.js. This project is an educational toolkit for creating, training and running compact GPT-style models client-side. It supports model creation, tokenisation, dataset preparation, training, and text generation with a single high-level entrypoint: `TeachableLLM`.

Live demo: https://lm.gen-ai.fi

**Design goals**

- Small models suitable for experimentation on laptops and mobile devices
- Clear, teachable APIs for training and generation
- Browser-first implementation using CPU, WebGL or WebGPU backends

## Installation

```bash
npm install @genai-fi/nanogpt
```

## Main concepts

- `TeachableLLM` — primary entrypoint. Create, load, save models; access training and responses APIs.
- `tokenise` — tokeniser helpers and token store (character and BPE tokenisers).
- `data` — helpers to load text data and stream conversational inputs.

The project export surface is centred around `TeachableLLM` (see `lib/main.ts`). This README focuses on the runtime API you will use in applications.

## Quick examples

Creating a model instance

```javascript
import { TeachableLLM } from '@genai-fi/nanogpt';

// Create a new model with a char or bpe tokeniser
const model = TeachableLLM.create('char', {
    vocabSize: 200,
    blockSize: 128,
    nLayer: 4,
    nHead: 4,
    nEmbed: 192,
});

// Switch backend if needed
await TeachableLLM.selectBackend('webgpu');
```

Training the tokeniser (when using streamed conversational data)

```javascript
import { data, tokenise } from '@genai-fi/nanogpt';

// Prepare streams using data.loadTextData or MemoryConversationStream
const streams = await data.loadTextData(['Some example text', 'More text']);

// Train the tokeniser on streams
const tokens = await model.trainTokeniser(streams);
console.log('Trained token count:', tokens);
```

Start a training job

```javascript
const job = await model.training.job(options, streams, datasets);

// Listen for training progress
model.training.on('progress', (job) => {
    console.log('Training job progress:', job.progress);
    console.log('Latest log entry:', job.history?.[job.history.length - 1]);
});

// Pause, resume, cancel via training API using the returned job id
```

Generate text (responses API)

```javascript
// Create a response — returns an id and may stream tokens via callback
const resp = await model.responses.create(
    {
        input: 'Once upon a time',
        maxLength: 100,
        temperature: 0.9,
    },
    (chunk) => {
        // called for intermediate chunks when provided
        console.log('Partial output:', chunk.output);
    }
);

console.log('Final output:', resp.output);

// Manage responses
// model.responses.cancel(id)
// model.responses.hook(id)
// model.responses.resume(id)
```

Tokenisers and token stores

```javascript
import { tokenise } from '@genai-fi/nanogpt';

// Character and BPE tokenisers are available
const { CharTokeniser, BPETokeniser, TokenStore, createTokenStore } = tokenise;

// Use TokenStore to persist prepared token sequences for training
```

Data helpers

```javascript
import { data } from '@genai-fi/nanogpt';

// Load plain text into conversation streams
const streams = await data.loadTextData(['Line one', 'Line two']);

// MemoryConversationStream is useful for in-memory conversations
const { MemoryConversationStream } = data;
```

## Development

Clone and install dependencies:

```bash
git clone https://github.com/knicos/genai-nanogpt.git
cd genai-nanogpt
npm install
```

Build and run browser tests:

```bash
npm run build
npm run dev
npm test
npm run test:gl
```

## Examples and demos

See the `browser-tests/` directory for small example pages demonstrating generation, training, and model loading.

## Acknowledgments

- Inspired by Andrej Karpathy's NanoGPT: https://github.com/karpathy/nanoGPT
- Built with TensorFlow.js: https://www.tensorflow.org/js
- Developed as part of the Finnish Generation AI research project: https://generation-ai-stn.fi

## Citation

If you use this library in your research, please cite:

```bibtex
@inproceedings{10.1145/3769994.3770061,
author = {Pope, Nicolas and Tedre, Matti},
title = {A Teachable Machine for Transformers},
year = {2025},
publisher = {Association for Computing Machinery},
doi = {10.1145/3769994.3770061},
booktitle = {Proceedings of the 25th Koli Calling International Conference on Computing Education Research},
}
```
