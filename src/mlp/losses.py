"""Loss functions used to train the NumPy neural network."""

import numpy as np


class SoftmaxCrossEntropy:
    """Compute mean cross-entropy from class logits and integer labels."""

    def __init__(self):
        """Initialize caches populated by the forward pass."""
        self.softmax = None
        self.targets = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute mean cross-entropy using numerically stable softmax.

        Args:
            logits: Unnormalized class scores with shape ``(batch, classes)``.
            targets: Integer class labels with shape ``(batch,)``.

        Returns:
            The mean cross-entropy loss for the batch.
        """
        self.targets = targets
        batch_size = logits.shape[0]

        # Subtracting the row maximum avoids overflow in the exponential.
        stable_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax = stable_exp / np.sum(stable_exp, axis=1, keepdims=True)

        self.softmax = softmax
        correct_probs = softmax[np.arange(batch_size), targets]

        # Compute the mean negative log-likelihood of the correct classes.
        loss = -np.mean(np.log(correct_probs + 1e-15))
        return float(loss)

    def backward(self) -> np.ndarray:
        """Return the gradient with respect to logits for the cached batch."""
        batch_size = self.softmax.shape[0]
        grad = self.softmax.copy()
        grad[np.arange(batch_size), self.targets] -= 1.0
        return grad / batch_size

