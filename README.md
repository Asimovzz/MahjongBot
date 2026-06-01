# MahjongBot SL

MahjongBot SL is a supervised-learning Mahjong bot for Chinese Official Mahjong. It converts Botzone match logs into policy-learning samples, trains a residual CNN policy network, and runs the trained policy through the Botzone text protocol. The submitted bot reached the top 40 on Botzone.

## Project Layout

```text
.
├── data/                 # Raw match logs and format notes
├── model/                # Checkpoints, including the provided model/8.pkl
├── src/mahjongbot/       # Package source code
│   ├── __main__.py       # Botzone runtime
│   ├── feature.py        # Rule-state tracking and feature extraction
│   ├── model.py          # CNN policy model
│   ├── preprocess.py     # Raw log -> .npz samples
│   ├── data_augment.py   # Suit-permutation augmentation
│   ├── dataset.py        # PyTorch dataset
│   └── supervised.py     # Training entry point
└── pyproject.toml
```

## Installation

Create an environment with Python 3.9+ and install the package in editable mode:

```bash
pip install -e .
```

The project depends on PyTorch and NumPy. Runtime feature extraction and preprocessing also require `MahjongGB`, which can be installed from the PyMahjongGB project:

```bash
pip install -e ".[rules]"
```

## Run The Bot

The default runtime checkpoint is `model/8.pkl`:

```bash
python -m mahjongbot --model model/8.pkl
```

After editable installation, the console command is also available:

```bash
mahjongbot --model model/8.pkl
```

## Data Pipeline

The raw data file is expected at `data/data.txt`. See `data/README.txt` for the match-log format.

Generate training samples:

```bash
mahjongbot-preprocess --input data/data.txt --output-dir data/preprocessed --count-path data/count.json
```

Generate suit-permutation augmentation:

```bash
mahjongbot-augment --count-path data/count.json --input-dir data/preprocessed --output-dir data/augmented
```

Train the policy:

```bash
mahjongbot-train --data-dir data --log-dir log --device cuda
```

Use `--augment` to train on `data/augmented` instead of `data/preprocessed`.

## Notes

- `data/data.txt` is large and should not be opened in a regular text editor.
- Generated samples, training logs, and private course documents are ignored by Git.
- The model masks invalid actions before selecting the policy output.
