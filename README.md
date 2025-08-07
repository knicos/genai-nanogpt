# GenAI NanoGPT

Developed as a part of the Finnish Generation AI research project. This is an implementation of [NanoGPT](https://github.com/karpathy/nanoGPT) for Tensorflow.js. It allows GPT models to be training and loaded within a web browser and exposes some XAI functionality.

Work in progress...

# Install

```
npm install @genai-fi/nanogpt
```

# Usage

```
import { TeachableLLM, CharTokeniser } from '@genai-fi/nanogpt';
import * as tf from '@tensorflow/tfjs';

const tokeniser = new CharTokeniser();
const model = TeachableLLM.create(tf, tokeniser, {
    vocabSize: 200,
    blockSize: 128,
    nLayer: 4,
    nHead: 3,
    nEmbed: 192,
    dropout: 0.0,
});
```
