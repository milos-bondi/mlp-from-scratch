# MLP from Scratch

A Multi-Layer Perceptron (MLP) implemented **from scratch using NumPy**, with manually implemented backpropagation, trained on the MNIST dataset and benchmarked against an equivalent **PyTorch** implementation.

> 🚧 Work in progress

## Objectives

- [ ] Implement `Linear` and `ReLU` layers with forward and backward passes
- [ ] Implement numerically stable softmax and cross-entropy loss
- [ ] Validate analytical gradients through numerical gradient checking
- [ ] Train the network on MNIST using mini-batch stochastic gradient descent (SGD)
- [ ] Reproduce the same architecture in PyTorch and compare performance

## Results

| Implementation | Test Accuracy | Time per Epoch |
|---|---|---|
| NumPy (from scratch) | – | – |
| PyTorch | – | – |

<!-- Add training loss plot: ![Loss](results/loss.png) -->

## Getting Started

Create a virtual environment, install the dependencies, and run the tests:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
```

## Project Structure

```text
src/mlp/       Core implementation (layers, losses, model, data, training)
tests/         Numerical gradient checks and environment tests
notebooks/     Exploratory analysis and visualizations
results/       Performance metrics and training plots
```

