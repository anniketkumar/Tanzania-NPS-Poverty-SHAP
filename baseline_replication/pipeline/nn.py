"""The neural-network base learner: a small PyTorch MLP wrapped by skorch.

skorch makes a PyTorch model behave like a scikit-learn estimator (fit/predict),
which is what lets it drop straight into the StackingClassifier alongside the
sklearn models. It's my stand-in for the deep-learning model the paper used.

A couple of things worth remembering here:
  * The network outputs raw scores (logits), and CrossEntropyLoss expects
    logits, so I don't put a softmax inside the model.
  * But the stacker wants real probabilities, so SoftmaxNeuralNetClassifier
    overrides predict_proba to run softmax on those logits. Without this I'd be
    feeding the meta-learner un-normalised scores.
  * torch is picky about dtypes, so there's a small cast-to-float32 step so the
    rest of the pipeline can keep passing ordinary float64 numpy arrays around.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier


class MLPModule(nn.Module):
    """A plain feed-forward net: input -> a few hidden layers -> class logits."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        hidden_dims: tuple[int, ...] = (32, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # raw logits


class SoftmaxNeuralNetClassifier(NeuralNetClassifier):
    """skorch classifier whose predict_proba returns true probabilities.

    As of skorch >= 0.12, the base NeuralNetClassifier.predict_proba()
    already applies softmax to the network's raw logits, so no manual
    softmax is needed.  This subclass exists only as a named type for
    clarity and to document the intent.
    """

    pass


def build_nn_classifier(
    input_dim: int,
    n_classes: int,
    *,
    hidden_dims: tuple[int, ...] = (32, 16),
    dropout: float = 0.1,
    max_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 0.0,
    device: str = "cpu",
    seed: int | None = None,
) -> SoftmaxNeuralNetClassifier:
    """Construct a ready-to-fit skorch classifier wrapping `MLPModule`."""
    if seed is not None:
        torch.manual_seed(seed)

    net = SoftmaxNeuralNetClassifier(
        module=MLPModule,
        module__input_dim=input_dim,
        module__n_classes=n_classes,
        module__hidden_dims=hidden_dims,
        module__dropout=dropout,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=weight_decay,
        lr=lr,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        train_split=None,        # the stacker already does the CV splitting, so
                                 # I don't want skorch holding out its own val set
        verbose=0,
    )
    return net


class FloatCastWrapper:
    """A tiny callable that casts arrays to float32 for the network.

    It lives inside a FunctionTransformer so the net can sit in a Pipeline next
    to sklearn models that use float64. I made it a real class (not a lambda)
    only because lambdas can't be pickled, and sklearn needs to clone/pickle the
    whole pipeline during cross-validation.
    """

    def __call__(self, X):
        return np.asarray(X, dtype=np.float32)
