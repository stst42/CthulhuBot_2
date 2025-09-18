# CthulhuBot_2
Repository for a Lovecraft-inspired character-level text generator built with Keras.

**Author**<br>
stst42

## Why this project exists
Being an R user, I wanted a reason to learn more Python and to explore deep learning through a fun problem. Generating pseudo-Lovecraft prose felt like the perfect crossover of interests.

## What changed recently?
All scripts are now fully commented and include docstrings so new readers can follow the training and generation pipeline without cross-referencing the source material. The documentation below was refreshed to match the current layout and workflow.

## Repository layout
```
scripts/
  train_apply_flow.py   # end-to-end training and sampling entrypoint
  models/
    first_lstm.py       # baseline, wider 2-layer LSTM
    second_lstm.py      # stacked LSTM used by default
    third_lstm.py       # small tweaks for A/B testing
weights/                # place checkpoints here (ignored by git)
```
The dataset is fetched on-the-fly from [urschrei/lovecraft](https://github.com/urschrei/lovecraft/blob/master/lovecraft.txt), so there is no local `data/` folder to keep in sync.

## Environment & tooling
- Python 3
- Keras + TensorFlow backend
- PyCharm Community Edition (author's IDE of choice)
- Linux Mint machine (AMD Ryzen 5 4000, NVIDIA GTX 1650, 32 GB RAM)

### Dev tooling
- Linter: Ruff (fast Python linter)
- Formatter: Black
- Hooks: pre-commit (runs Ruff + Black on commit and push)
- Config: `.ruff.toml`

Setup:
1. Create/activate your virtualenv
2. Install dev deps: `pip install -r requirements-dev.txt`
3. Install git hooks: `pre-commit install --hook-type pre-commit --hook-type pre-push`

Common tasks:
- Lint: `make lint` (or `ruff check .`)
- Format: `make format` (or `black --line-length 100 .`)

## Training and sampling
1. Ensure a writable `weights/` directory exists alongside `scripts/`.
2. Run `python scripts/train_apply_flow.py`.
   - The script downloads the corpus, prepares the dataset, logs progress, trains the model, and immediately samples new text.
   - Model checkpoints are saved to `weights/weights-improvement-v4-*.hdf5` via `ModelCheckpoint`.
   - Basic logging and defensive error handling will exit cleanly if downloading, training, or prediction fails.

The script now contains inline explanations for each stage (data prep, model definition, training, and generation) and a small `execfile` compatibility shim so it can run under Python 3 while keeping the original architecture snippets intact.

## Results snapshot
With a temperature of `1.7` and the seed `"the creature in the darkness"`, the model produces surreal but thematically consistent prose. Raising the temperature emphasises weirdness; lowering it makes the output more conservative and repetitive.

## References
- [LSTM text generation walkthrough](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
- [TensorFlow GPU compatibility discussion](https://forums.developer.nvidia.com/t/does-gtx-1050ti-or-1650-for-notebook-support-tensorflow-gpu/77384/9)
- [StackOverflow thread on improving generation diversity](https://stackoverflow.com/questions/47125723/keras-lstm-for-text-generation-keeps-repeating-a-line-or-a-sequence)
