# MLP from Scratch

A multi-layer perceptron implemented with NumPy, including manual forward and
backward passes, and trained on the MNIST handwritten-digit dataset. The
repository also includes exploratory notebooks and a small PyTorch-based
gradient comparison.

## Features

- Fully connected and ReLU layers with forward and backward passes
- Numerically stable softmax cross-entropy loss
- Mini-batch stochastic gradient descent training
- MNIST exploration and NumPy warm-up notebooks
- Training metrics and sample prediction visualizations

## Getting Started

Python 3.10 or newer is required. From the repository root, install the project
and its dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the training script from the repository root:

```bash
python -m mlp.train
```

The script downloads MNIST to `data/` on its first run and saves visualizations
to `results/`.

## Development

Install the development dependencies with `uv` and run the test suite with:

```bash
uv sync --dev
uv run pytest
```

## Project Structure

```text
src/mlp/       NumPy implementation (data, layers, losses, model, training)
tests/         Gradient, model, and environment checks
notebooks/     NumPy exercises and MNIST exploration
results/       Generated training visualizations (ignored by Git)
data/          Downloaded MNIST data (ignored by Git)
```
