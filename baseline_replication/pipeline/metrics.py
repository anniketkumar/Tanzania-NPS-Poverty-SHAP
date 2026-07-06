"""All the evaluation metrics in one place.

For each run I compute the usual classification scores: accuracy, and the
macro-averaged precision / recall / F1 / Jaccard (macro = average the per-class
scores equally, so the smaller cluster counts as much as the larger one). I also
report Cohen's kappa and Matthews correlation, which both discount agreement you
would have got by chance, and a macro one-vs-one ROC AUC from the predicted
probabilities. The confusion matrix comes along too.

Note: `gmean` here is sqrt(precision * recall), i.e. the geometric mean of those
two macro scores. That's a slightly non-standard "G-mean" (the more common one
is the geometric mean of the per-class recalls), so read it as just another
balanced summary of precision and recall rather than the textbook definition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def multiclass_auc(y_true, proba, classes) -> float:
    """Macro one-vs-one AUC. I one-hot the true labels (in the same class order
    as the probability columns) so roc_auc_score can line them up correctly."""
    y_true_bin = pd.get_dummies(pd.Categorical(y_true, categories=classes))
    y_true_bin = y_true_bin[classes]
    return roc_auc_score(
        y_true_bin.values, proba, average="macro", multi_class="ovo"
    )


def compute_metrics_all(y_true, y_pred, proba, classes) -> dict:
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    jaccard = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    gmean = float(np.sqrt(recall * precision))
    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc_macro = float(multiclass_auc(y_true, proba, classes))
    except Exception:
        auc_macro = np.nan
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "kappa": kappa,
        "mcc": mcc,
        "gmean": gmean,
        "auc_macro": auc_macro,
    }


def per_class_metrics(y_true, y_pred) -> pd.DataFrame:
    labels = np.unique(y_true)
    recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return pd.DataFrame(
        {
            "label": labels,
            "recall": recalls,
            "precision": precisions,
            "f1": f1s,
        }
    )


def confusion(y_true, y_pred) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred).tolist()
