"""Train the NumPy MLP on MNIST and save evaluation visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mlp.data import load_mnist
from mlp.layers import Linear, ReLU
from mlp.losses import SoftmaxCrossEntropy
from mlp.model import MLP

from time import perf_counter


random_seed = 43
rng = np.random.default_rng(seed=random_seed)
INPUT_FEATURES = 784
HIDDEN_FEATURES = 128
OUTPUT_FEATURES = 10
batch_size = 64

epochs = 10


def main():
    """Train the model, report epoch metrics, and save result plots."""

    initial_time = perf_counter()

    X_train, y_train, X_test, y_test = load_mnist()

    linear_layer_1 = Linear(in_features=INPUT_FEATURES, out_features=HIDDEN_FEATURES)
    activation = ReLU()
    linear_layer_2 = Linear(in_features=HIDDEN_FEATURES, out_features=OUTPUT_FEATURES)

    model = MLP(layers=[linear_layer_1, activation, linear_layer_2])
    loss_function = SoftmaxCrossEntropy()

    history_loss = []
    accuracy = []

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(X_train))
        shuffled_X = X_train[indices]
        shuffled_y = y_train[indices]

        batch_losses = []
        for start in range(0, len(X_train), batch_size):
            X_batch = shuffled_X[start : start + batch_size]
            y_batch = shuffled_y[start : start + batch_size]

            logits = model.forward(x=X_batch)
            loss = loss_function.forward(logits=logits, targets=y_batch)
            gradient = loss_function.backward()
            model.backward(gradient)
            model.step(lr=0.1)

            batch_losses.append(loss)

        mean_loss = float(np.mean(batch_losses))
        history_loss.append(mean_loss)

        test_logits = model.forward(x=X_test)
        preds = np.argmax(test_logits, axis=1)
        test_acc = np.mean(preds == y_test)

        accuracy.append(test_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | Loss: {mean_loss:.4f} "
            f"| Test Acc: {test_acc:.4f}"
        )

    total_time = perf_counter() - initial_time

    print(f"Total pipeline time: {total_time}")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    results_dir = PROJECT_ROOT / "results"

    plot_metrics(
        losses=history_loss,
        accuracies=accuracy,
        save_path=results_dir / "metrics.png",
    )
    plot_sample_prediction(
        model,
        X_test,
        y_test,
        save_path=results_dir / "sample_prediction.png",
    )




def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities with a numerically stable softmax."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def plot_sample_prediction(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    idx: int | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot one test image beside the model's predicted class probabilities.

    Args:
        model: Trained MLP used to predict the sample's class probabilities.
        X_test: Flattened test images with shape ``(N, 784)``.
        y_test: Integer labels corresponding to ``X_test``.
        idx: Test-set index to display. A random index is selected if omitted.
        save_path: Optional path for saving the figure. If omitted, display it.
    """
    if idx is None:
        idx = int(np.random.randint(0, len(X_test)))

    sample_x = X_test[idx : idx + 1]  # Keep the batch dimension: (1, 784).
    true_label = int(y_test[idx])

    logits = model.forward(sample_x)
    probs = softmax(logits)[0]  # One probability for each of the 10 classes.
    pred_label = int(np.argmax(probs))

    # Highlight the true and predicted classes; leave the other bars gray.
    colors = ["lightgray"] * 10
    colors[true_label] = "steelblue"
    colors[pred_label] = "seagreen" if pred_label == true_label else "crimson"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Show the 28-by-28 input image.
    ax1.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    ax1.set_title(f"Index #{idx} | True: {true_label} | Predicted: {pred_label}")
    ax1.axis("off")

    # Show the predicted probability for each digit class.
    classes = np.arange(10)
    bars = ax2.bar(classes, probs * 100, color=colors, edgecolor="black")
    ax2.set_xticks(classes)
    ax2.set_ylim(0, 108)
    ax2.set_xlabel("Class (digit)")
    ax2.set_ylabel("Probability (%)")
    ax2.set_title(f"Model confidence: {probs[pred_label] * 100:.2f}%")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, p in zip(bars, probs):
        if p > 0.01:  # Label bars above 1% for readability.
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                p * 100 + 2,
                f"{p * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    plt.tight_layout()

    if save_path is not None:
        output_file = Path(save_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_metrics(
    losses: list[float],
    accuracies: list[float],
    save_path: str | Path = "results/metrics.png",
) -> None:
    """Plot training loss and test accuracy side by side and save the figure.

    Args:
        losses: Mean training loss recorded after each epoch.
        accuracies: Test accuracy recorded after each epoch.
        save_path: Destination path for the generated figure.
    """
    output_file = Path(save_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Plot training loss by epoch.
    ax1.plot(
        epochs,
        losses,
        marker="o",
        linewidth=2,
        color="steelblue",
        label="Train Loss",
    )
    ax1.set_title("Training Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Plot test accuracy as a percentage.
    acc_pct = [a * 100 for a in accuracies]
    ax2.plot(
        epochs,
        acc_pct,
        marker="o",
        linewidth=2,
        color="seagreen",
        label="Test Accuracy",
    )
    ax2.set_title("Test Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(epochs)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
