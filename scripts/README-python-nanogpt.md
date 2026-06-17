# Python NanoGPTV2 Tools

This folder contains a Python implementation of the GenAI NanoGPT v2 architecture used by this repository, with compatibility for the same model archive layout:

- `config.json`
- `tokeniser.json`
- `model.safetensors` (the repository's custom safetensors variant)
- `meta.json`

## Files

- `nanogpt_py.py`: CLI entrypoint
- `py_nanogpt/model.py`: model architecture and generation
- `py_nanogpt/tokenizer.py`: char/BPE tokenizer compatibility
- `py_nanogpt/io.py`: zip and custom safetensors load/save

## Install

```bash
pip install torch numpy
```

## Usage

### Use a JSON config file for defaults

The CLI always loads `scripts/nanogpt.defaults.json` as the default source of truth.

You can avoid repeating long command lines by storing common defaults there.

Command-line arguments always override config values.

Optionally, pass `--config` to overlay another JSON config on top of `nanogpt.defaults.json`.

Training defaults now run until terminated. Set `steps` or `max_epochs` in config (or `--steps` / `--max-epochs` on CLI) if you want a fixed cap.

If training stops earlier than expected, check `train.steps` and `train.max_epochs` in `scripts/nanogpt.defaults.json` (both should be `0` for unbounded training).

Example config (`scripts/nanogpt.defaults.json`):

```json
{
    "global": {
        "quiet": true,
        "progress_interval": 500
    },
    "train": {
        "model": "models/base.zip",
        "text_dirs": ["data/pretrain"],
        "output": "models/pretrained.zip",
        "batch_size": 12,
        "learning_rate": 0.0003,
        "metrics": "accuracy,perplexity,learningRate"
    },
    "prepare-data": {
        "model": "models/base.zip",
        "text_dirs": ["data/pretrain"],
        "output_token_ids_file": "artifacts/train_tokens.pt"
    },
    "generate": {
        "model": "models/pretrained.zip",
        "max_new_tokens": 120,
        "temperature": 0.9,
        "top_p": 0.9,
        "only_new_text": true
    }
}
```

Run with config:

```bash
python scripts/nanogpt_py.py train
python scripts/nanogpt_py.py prepare-data
python scripts/nanogpt_py.py generate --prompt "Hello"
```

Run with an overlay config:

```bash
python scripts/nanogpt_py.py --config scripts/nanogpt.local.json train
```

Notes:

- Key names can use either `snake_case` (for example `output_token_ids_file`) or CLI-style `kebab-case` (for example `output-token-ids-file`).
- You can define command sections either directly at top level (`"train"`, `"prepare-data"`, `"generate"`) or under a `"commands"` object.
- Unknown config keys are ignored with a warning.

### Prepare tokenizer + tokenized dataset files

Use `prepare-data` to build and save:

- tokenizer file (`tokeniser.json`-compatible)
- pretokenized ids file (`.pt` tensor payload)

This lets future training runs skip tokenization entirely.

You can optionally provide a base model and reuse its tokenizer/vocab directly.

```bash
python scripts/nanogpt_py.py prepare-data \
  --tokenizer-type bpe \
  --vocab-size 4096 \
  --text-dirs data/pretrain \
  --output-tokenizer-file artifacts/tokeniser.json \
  --output-token-ids-file artifacts/train_tokens.pt
```

Reuse tokenizer from an existing model zip:

```bash
python scripts/nanogpt_py.py prepare-data \
  --model models/base.zip \
  --text-dirs data/pretrain \
  --output-token-ids-file artifacts/base_tokens.pt
```

Optional: if you also want a standalone tokenizer JSON emitted, include `--output-tokenizer-file`.

When `--model` is provided, tokenizer-building flags such as `--tokenizer-type`, `--vocab-size`, and tokenizer vocab/merges files are ignored.

Supported input formats (TS-aligned subset):

- `.txt`: whole file as one text record
- `.csv`: one record per row from a selected column
- `.json`: expected to be an array; each item maps to one text record
- `.jsonl`: one record per line; supports text items and turn-based conversations

Parquet support note:

- Direct parquet loading is intentionally not in the TypeScript browser loader.
- Use `scripts/parquet_to_jsonl.py` to convert parquet into TS-compatible JSONL (conversation-array-per-line format).
- Non-conversational rows are emitted using the special `text` role:
    - `[ { "role": "text", "content": "..." } ]`

Parquet conversion examples:

```bash
python scripts/parquet_to_jsonl.py data/input.parquet -o data/output.jsonl
python scripts/parquet_to_jsonl.py data/input.parquet -o data/output.jsonl --zip
python scripts/parquet_to_jsonl.py data/chat.parquet --conversation-column conversation --zip
```

Parquet converter install dependency:

```bash
pip install pyarrow
```

CSV rules:

- Default column name is `text` (use `--csv-column` to override)
- Header detection uses first-row heuristic (all cells length < 64)
- CSV field-size limit is raised automatically to support very large text cells
- Override header behavior with:
    - `--csv-has-header`
    - `--csv-no-header`

JSON rules:

- File must be a JSON array
- Each item is converted to text as:
    - string -> the string itself
    - object with `text` -> that field
    - otherwise -> JSON-stringified item

JSONL rules:

- Empty lines are skipped
- If a line parses to a conversation array of `{ role, content }`, it is treated as a turn-based conversation record
- Otherwise line is converted like JSON item mapping above
- If a line cannot be parsed as JSON, the raw line is used as text

Conversation tokenization behavior:

- Turn-based conversation records are tokenized with role wrappers:
    - `<|user_start|> ... <|user_end|>`
    - `<|assistant_start|> ... <|assistant_end|>`
    - `<|system_start|> ... <|system_end|>`
- Each conversation record is wrapped with BOS/EOS tokens
- This applies in preprocessing and pretraining, as requested

Char tokenizer example:

```bash
python scripts/nanogpt_py.py prepare-data \
  --tokenizer-type char \
  --vocab-size 2048 \
  --text-files data/wiki.txt data/books.txt \
  --output-tokenizer-file artifacts/char_tokeniser.json \
  --output-token-ids-file artifacts/char_tokens.pt
```

### Terminal progress logging

The training command now prints staged progress for:

- loading each data source
- automatic BPE vocabulary/merge construction
- corpus tokenization progress
- final training setup counts
- step logs including `train_loss`, `val_loss`, and `samples_per_second` (context slices/second)

Control verbosity with:

- `--progress-interval`: update frequency for preprocessing progress (default `1000`)
- `--quiet`: suppress non-essential preprocessing logs

Manual checkpoint controls during training:

- Type `save` then press Enter: save a checkpoint immediately and continue.
- Type `stop` then press Enter: save a checkpoint and stop training.
- Press `Ctrl+C` (or send `SIGTERM`): save a checkpoint after the current step and stop.
- Periodic best-checkpoint mode: set `--checkpoint-interval N` to evaluate every N steps and overwrite `--output` only when `val_loss` improves vs the last saved checkpoint.

Use `--no-stdin-commands` if you want to disable typed stdin commands.

Validation split behavior:

- Validation is controlled by `--validation-split` (default `0.1`).
- At each log step, validation loss is computed on validation batches (`--validation-batches`) and printed as `val_loss`.
- If no validation set is available (for example `--validation-split 0`), logs print `val_loss=nan`.

Example:

```bash
python scripts/nanogpt_py.py train \
  --init-from-scratch \
  --tokenizer-type bpe \
  --vocab-size 4096 \
  --text-dirs data/pretrain \
  --output models/new-bpe-auto.zip \
  --progress-interval 500
```

Performance note:

- Automatic BPE building now uses an incremental pair-count update strategy aligned with the TypeScript tokenizer logic (instead of recomputing all pair counts each merge), which significantly improves speed on larger corpora.

### Train a new model from scratch (no input zip)

You can initialize a brand new NanoGPTV2 during training by using `--init-from-scratch`.

Minimal example:

```bash
python scripts/nanogpt_py.py train \
  --init-from-scratch \
  --text-file data/train.txt \
  --output models/new-from-scratch.zip \
  --n-layer 6 \
  --n-head 6 \
  --n-embed 384 \
  --block-size 256 \
  --mlp-factor 4 \
  --vocab-size 2048 \
  --tokenizer-type char \
  --steps 5000 \
  --batch-size 16
```

How this works:

- The script builds a new model from the architecture flags.
- For `--tokenizer-type char`, it builds vocabulary from your training text (most common chars, plus special tokens).
- Then it immediately starts pretraining and saves a normal model zip.

### Train a new model from scratch with BPE tokenizer

If you do not provide a vocab file, BPE vocab and merges are built automatically from the provided training data.

Automatic BPE build example:

```bash
python scripts/nanogpt_py.py train \
  --init-from-scratch \
  --text-dirs data/pretrain \
  --output models/new-bpe-auto.zip \
  --n-layer 8 \
  --n-head 8 \
  --n-embed 512 \
  --block-size 256 \
  --mlp-factor 4 \
  --tokenizer-type bpe \
  --vocab-size 4096 \
  --steps 8000 \
  --batch-size 12
```

You can also provide an existing vocab/merges payload.

```bash
python scripts/nanogpt_py.py train \
  --init-from-scratch \
  --text-dirs data/pretrain \
  --output models/new-bpe.zip \
  --n-layer 8 \
  --n-head 8 \
  --n-embed 512 \
  --block-size 256 \
  --mlp-factor 4 \
  --tokenizer-type bpe \
  --tokenizer-vocab-file tokenizers/my_tokeniser.json \
  --steps 8000 \
  --batch-size 12
```

Notes:

- `--tokenizer-vocab-file` accepts either:
    - a JSON list of vocab tokens, or
    - a `tokeniser.json` object containing `vocab` (and optionally `merges`).
- If merges are stored separately, pass `--tokenizer-merges-file`.
- When `--tokenizer-type bpe` is used with no vocab file, `--vocab-size` controls the target vocabulary size for automatic BPE training.

### Validate/load a model zip

```bash
python scripts/nanogpt_py.py load --model path/to/model.zip
```

### Re-save a model zip

```bash
python scripts/nanogpt_py.py save --model path/to/model.zip --output path/to/out.zip
```

### Fine-tune on text

```bash
python scripts/nanogpt_py.py train \
  --model path/to/model.zip \
  --text-file path/to/text.txt \
  --output path/to/finetuned.zip \
  --batch-size 8
```

### Train from pretokenized data (skip tokenization)

If you already ran `prepare-data`, pass the token ids directly:

```bash
python scripts/nanogpt_py.py train \
  --model models/base.zip \
  --token-ids-file artifacts/train_tokens.pt \
  --tokenizer-file artifacts/tokeniser.json \
  --output models/finetuned-faststart.zip \
  --batch-size 16
```

Notes:

- `--tokenizer-file` is optional for existing model zips but recommended for validation.
- For `--init-from-scratch`, text input is still required to build the initial model/tokenizer.
- By default, training runs until manually stopped (`stop` command, `Ctrl+C`, or `SIGTERM`).
- To cap duration, set `--steps N` or `--max-epochs N`.

### Pre-train from multiple data sources

Use any combination of:

- `--text-file`: one file
- `--text-files`: many specific files
- `--text-dirs`: recursively include `*.txt`
- `--globs`: glob patterns from current working directory

```bash
python scripts/nanogpt_py.py train \
  --model models/base.zip \
  --text-files data/wiki.txt data/books.txt data/code.txt \
  --text-dirs data/corpus_a data/corpus_b \
  --globs "datasets/**/*.txt" "samples/*.txt" \
  --output models/pretrained-mix.zip \
  --batch-size 16
```

### Full pretraining options example

This command exercises nearly all TS-relevant training options that are meaningful for this Python pretraining path:

```bash
python scripts/nanogpt_py.py train \
  --model models/base.zip \
  --text-dirs data/pretrain \
  --output models/pretrained-full.zip \
  --steps 0 \
  --max-epochs 3 \
  --batch-size 12 \
  --validation-split 0.1 \
  --validation-batches 12 \
  --context-scaling 0.75 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-5 \
  --warmup-steps 1000 \
  --decay-epochs 100 \
  --epoch-steps 0 \
  --beta1 0.9 \
  --beta2 0.99 \
  --epsilon 1e-8 \
  --weight-decay 0.1 \
  --loss-scaling 1.0 \
  --clip-norm 1.0 \
  --gradient-checkpointing \
  --mixed-precision \
  --mixed-precision-dtype auto \
  --enable-tf32 \
  --compile-model \
  --compile-mode reduce-overhead \
  --optimizer-impl auto \
  --attention-backend auto \
  --label-smoothing 0.05 \
  --dropout 0.1 \
  --layer-drop 0.05 \
  --trainable-weights "*" \
  --metrics "accuracy,perplexity,gradientNorm,tokensPerSecond,learningRate" \
  --prompt "The future of browser-native AI is" \
  --log-interval 20 \
  --checkpoint-interval 200
```

### Train only selected weights (TS `trainableWeights` parity)

```bash
python scripts/nanogpt_py.py train \
  --model models/base.zip \
  --text-file data/domain.txt \
  --output models/domain-head-only.zip \
  --steps 1200 \
  --trainable-weights "token_embedding" "blocks.*.attn.c_proj" "blocks.*.mlp.mlp_out"
```

### CPU-only deterministic-ish debug run

```bash
python scripts/nanogpt_py.py train \
  --model models/base.zip \
  --text-file data/tiny.txt \
  --output models/debug.zip \
  --steps 100 \
  --batch-size 2 \
  --context-scaling 0.5 \
  --cpu
```

## Training Option Parity vs TypeScript

Implemented and active in Python pretraining:

- `batchSize` -> `--batch-size`
- `maxEpochs` -> `--max-epochs` (used when `--steps <= 0`)
- `logInterval` -> `--log-interval`
- `prompt` -> `--prompt`
- `validationSplit` -> `--validation-split`
- `gradientCheckpointing` -> `--gradient-checkpointing`
- `mixedPrecision` -> `--mixed-precision`
- Mixed precision/runtime knobs:
    - `--mixed-precision-dtype` (`auto|bf16|fp16`)
    - `--enable-tf32` / `--disable-tf32`
- Compile/runtime backends:
    - `--compile-model` / `--compile-mode`
    - `--optimizer-impl` (`auto|fused|foreach|default`)
    - `--attention-backend` (`auto|sdpa|math`)
- `trainableWeights` -> `--trainable-weights`
- `metrics` -> `--metrics`
- `contextScaling` -> `--context-scaling`
- `labelSmoothing` -> `--label-smoothing`
- `dropout` -> `--dropout`
- `layerDrop` -> `--layer-drop`
- Optimizer/scheduler options from `AdamWOptimizerConfig`:
    - `learningRate` -> `--learning-rate`
    - `beta1` -> `--beta1`
    - `beta2` -> `--beta2`
    - `epsilon` -> `--epsilon`
    - `weightDecay` -> `--weight-decay`
    - `lossScaling` -> `--loss-scaling`
    - `clipNorm` -> `--clip-norm`
    - `warmupSteps` -> `--warmup-steps`
    - `decayEpochs` -> `--decay-epochs`
    - `minLearningRate` -> `--min-learning-rate`
    - `epochSteps` -> `--epoch-steps`

Accepted for compatibility but not used in Python pretraining (TS SFT/LoRA-specific):

- `sftMode` -> `--sft-mode`
- `maskedLoss` -> `--masked-loss`
- `loraConfig` -> `--lora-config`
- `loraName` -> `--lora-name`

Not currently implemented in this Python script:

- `orthoGrad` optimizer behavior
- `gradientStatistics`, `weightNorm`, `weightStatistics`, `memoryUsage` metrics

### Generate

```bash
python scripts/nanogpt_py.py generate \
  --model path/to/model.zip \
  --prompt "Hello" \
  --max-new-tokens 120 --temperature 0.9 --top-p 0.95
```
