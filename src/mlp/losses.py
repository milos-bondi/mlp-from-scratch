"""Softmax + cross-entropy combinate (numericamente stabili)."""
import numpy as np


class SoftmaxCrossEntropy:
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """logits: (batch, classes), targets: (batch,) interi. Ritorna la loss media.

        Suggerimento: sottrai il max per riga prima di exp() per evitare overflow.
        """
        raise NotImplementedError("TODO")

    def backward(self) -> np.ndarray:
        """Ritorna dL/dlogits, shape (batch, classes). Risultato: (softmax - one_hot) / batch"""
        raise NotImplementedError("TODO")
