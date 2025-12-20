# GenAI NanoGPT

A browser-native implementation of GPT language models built on TensorFlow.js, developed as part of the Finnish Generation AI research project. This library enables training, fine-tuning, and inference of transformer-based language models entirely in the browser with support for explainable AI (XAI) features. It is intended to be used as an educational tool for learning about the model training process since it targets mostly tiny models. In principle it could be adapted to load other pre-trained models from Hugging Face.

## Overview

GenAI NanoGPT is inspired by [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT) but reimagined for the browser using TensorFlow.js. It provides a complete pipeline for:

-   **Training** language models from scratch in the browser
-   **Loading** pre-trained models from various sources (Hugging Face, local files)
-   **Generating** text efficiently on a wide range of devices
-   **Analyzing** model behavior through attention visualization and embeddings
-   **Optimizing** performance across CPU, WebGL, and WebGPU backends

### Key Features

-   🚀 **Browser-Native**: No server required - train and run models entirely client-side
-   📱 **Works on Small Devices**: Train models on iPads, phones, and Chromebooks - no powerful hardware needed
-   🎯 **Multiple Backends**: Automatic backend selection (CPU, WebGL, WebGPU) for optimal performance
-   🔧 **Flexible Tokenization**: Support for both character-level and BPE tokenizers
-   📊 **XAI Support**: Attention score visualization, gradient analysis, and embedding extraction
-   💾 **Model Persistence**: Save and load models in SafeTensors format
-   ⚡ **Performance Optimizations**: Custom WebGPU kernels, gradient checkpointing, and mixed precision training
-   🎨 **Real-time Training**: Live training metrics and generation during training

## Installation

```bash
npm install @genai-fi/nanogpt
```

## Quick Start

### Creating and Training a Model

```javascript
import { TeachableLLM, selectBackend } from '@genai-fi/nanogpt';

// Select the best available backend
await selectBackend('webgpu'); // or 'webgl', 'cpu'

// Create a new model
const model = TeachableLLM.create('char', {
    vocabSize: 200,
    blockSize: 128, // Context window size
    nLayer: 4, // Number of transformer layers
    nHead: 4, // Number of attention heads
    nEmbed: 192, // Embedding dimension
    dropout: 0.1,
    useRope: true, // Use Rotary Position Embeddings
});

// Training data
const trainingText = [
    'The quick brown fox jumps over the lazy dog.',
    'A journey of a thousand miles begins with a single step.',
    // ... more text
];

// Train the model
await model.train(trainingText, {
    batchSize: 16,
    learningRate: 3e-4,
    maxSteps: 1000,
    logInterval: 10,
    validationSplit: 0.1,
});

// Generate text
const output = await model.generateText('Once upon a time', {
    maxLength: 100,
    temperature: 0.8,
    topP: 0.9,
});

console.log(output);
```

### Loading a Pre-trained Model

```javascript
import { TeachableLLM, waitForModel } from '@genai-fi/nanogpt';

// Load from Hugging Face
const model = TeachableLLM.loadModel('username/model-name');

// Or load from a file
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    const model = TeachableLLM.loadModel(file);
    await waitForModel(model);

    const text = await model.generateText('Hello');
    console.log(text);
});
```

## Event Handlers and Real-time Updates

### Monitoring Training Progress

Track training metrics in real-time with event handlers:

```javascript
const model = TeachableLLM.create('char', config);

// Listen for training step updates
model.on('trainStep', (step, progress) => {
    console.log(`Step ${step.step}/${progress.totalSteps}`);
    console.log(`Loss: ${step.loss.toFixed(4)}`);
    console.log(`Validation Loss: ${step.valLoss?.toFixed(4) || 'N/A'}`);
    console.log(`Progress: ${(progress.progress * 100).toFixed(1)}%`);
    console.log(`Time Remaining: ${progress.timeRemaining}s`);

    // Update UI progress bar
    updateProgressBar(progress.progress);
    updateLossChart(step.loss, step.valLoss);
});

await model.train(trainingText, options);
```

### Real-time Token Generation

Stream generated tokens as they're produced:

```javascript
const generator = model.generator();

// Listen for generated tokens
generator.on('tokens', (tokens) => {
    // tokens is an array of new token IDs
    const text = model.tokeniser.decode(tokens);
    console.log('New tokens:', text);

    // Update UI incrementally
    appendToOutput(text);
});

// Generation lifecycle events
generator.on('start', () => {
    console.log('Generation started');
    showSpinner();
});

generator.on('stop', () => {
    console.log('Generation complete');
    hideSpinner();
});

generator.on('error', (error) => {
    console.error('Generation error:', error);
});

// Start generation
await generator.generate('Once upon a time', {
    maxLength: 200,
    temperature: 0.8,
});
```

## Training on Small Devices

GenAI NanoGPT is designed to work efficiently on resource-constrained devices like iPads, phones, and Chromebooks:

### Recommended Settings for Small Devices

```javascript
// Smaller model configuration for mobile devices
const mobileModel = TeachableLLM.create('char', {
    vocabSize: 200,
    blockSize: 128, // Smaller context window
    nLayer: 4, // Fewer layers
    nHead: 3, // Fewer attention heads
    nEmbed: 192, // Smaller embeddings
});

// Training options optimized for limited memory
await mobileModel.train(trainingText, {
    batchSize: 8, // Smaller batch size
    learningRate: 3e-4,
    maxSteps: 500,
    validationSplit: 0.1,
    logInterval: 50,
    gradientCheckpointing: true,
    mixedPrecision: true,
});
```

### Tips for Training on Mobile Devices

1. **Start Small**: Use smaller models (4 layers) and shorter context windows (128 tokens)
2. **Reduce Batch Size**: Use batch sizes of 8-16 depending on available memory
3. **Use Character Tokenization**: Character-level tokenizers use less memory than BPE
4. **Optimize Training Data**: Use smaller datasets or train in stages

## Advanced Usage

### Attention Visualization

```javascript
const generator = model.generator();

const text = await generator.generate('Prompt', {
    attentionScores: true,
    maxLength: 50,
});

// Get attention data for visualization
const attentionData = generator.getAttentionData();
// Shape: [num_tokens][num_layers][num_heads][seq_len][seq_len]

const probabilities = generator.getProbabilitiesData();
// Shape: [num_tokens][seq_len][vocab_size]
```

### Streaming Generation

```javascript
const generator = model.generator();

generator.on('tokens', (tokens) => {
    // Update UI with new tokens in real-time
    updateDisplay(tokens);
});

generator.on('start', () => console.log('Generation started'));
generator.on('stop', () => console.log('Generation complete'));

await generator.generate('Once upon a time', {
    maxLength: 200,
});
```

### Memory Management

```javascript
// Enable profiling
model.enableProfiler = true;

// After training/generation
const profiler = model.getProfiler();
if (profiler) {
    console.log('Memory stats:', profiler.getStats());
}

// Clean up
model.dispose();
```

## Examples

See the [`browser-tests`](browser-tests/) directory for complete examples:

-   [`generate.html`](browser-tests/generate.html): Text generation with UI
-   [`rope-train.html`](browser-tests/rope-train.html): Training a model with RoPE
-   [`hf.html`](browser-tests/hf.html): Loading from Hugging Face
-   [`loader.html`](browser-tests/loader.html): Loading different file formats
-   [`perf.html`](browser-tests/perf.html): Performance testing

## Development

### Setup

```bash
git clone https://github.com/knicos/genai-nanogpt.git
cd genai-nanogpt
npm install
```

### Building

```bash
npm run build       # Build for production
npm run dev         # Development mode with watch
```

### Testing

```bash
npm test            # Run all tests
```

### Browser Tests

```bash
npm run test:gl       # Start dev server
```

### Project Structure

```
lib/
├── models/          # Model architectures (NanoGPT)
├── layers/          # Transformer layers (attention, MLP, etc.)
├── ops/             # Custom TensorFlow.js operations
│   ├── cpu/         # CPU kernels
│   ├── webgl/       # WebGL kernels
│   └── webgpu/      # WebGPU kernels
├── training/        # Training utilities and optimizers
├── tokeniser/       # Tokenization implementations
├── loader/          # Model loading/saving
├── utilities/       # Helper functions
└── TeachableLLM.ts  # Main API
```

### Custom Operations

This library implements several custom TensorFlow.js operations optimized for transformer models:

-   **RoPE**: Rotary Position Embeddings
-   **Attention Mask**: Causal attention masking
-   **RMS Norm**: Root Mean Square normalization
-   **Adam Optimizer**: Extended Adam with weight decay
-   **16-bit Operators**: To enable mixed-precision training

See [`lib/ops`](lib/ops/) for implementations.

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Style

This project uses ESLint and Prettier for code formatting:

```bash
npm run lint        # Check code style
```

## Performance Tips

1. **Use WebGPU**: Provides the best performance for training and inference
2. **Batch Size**: Larger batches improve GPU utilization but require more memory
3. **Mixed Precision**: Enable for faster training on supported hardware (coming soon)
4. **Gradient Checkpointing**: Reduce memory usage during training, but slower
5. **Use RoPE**: More efficient than absolute position embeddings
6. **Start Small on Mobile**: Use 2-4 layers and batch size 2-8 on phones/tablets

## Acknowledgments

-   Inspired by [Andrej Karpathy's NanoGPT](https://github.com/karpathy/nanoGPT)
-   Built with [TensorFlow.js](https://www.tensorflow.org/js)
-   Developed as part of the Finnish [Generation AI research project](https://generation-ai-stn.fi)

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
