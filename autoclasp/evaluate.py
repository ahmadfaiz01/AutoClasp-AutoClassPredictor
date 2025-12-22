import time
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import plotly.graph_objs as go


def get_probability(model: Any, X):
    """
    Get probability / score outputs for ROC-AUC.

    Tries:
    1. model.predict_proba(X)
    2. model.decision_function(X)
    Returns None if neither works.
    """
    # Try probability estimates
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            return np.asarray(proba)
        except Exception:
            return None

    # Try decision scores
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
            return np.asarray(scores)
        except Exception:
            return None


# Funtion for plotting confusion matrix
def plot_confusion_matrix(confusion_matrix: np.ndarray, labels=None, title: str = "Confusion Matrix") -> go.Figure:

    # Converting matrix into list of lists for Plotly
    z = confusion_matrix.tolist()

    # Axis labels
    # If labels are available use them
    if labels is not None:
        x = labels
        y = labels
    else:
        # x-axis: one label per predicted class which are in matrix columns
        x = list(range(confusion_matrix.shape[1]))

        # y-axis: one label per true class which are in matrix rows
        y = list(range(confusion_matrix.shape[0]))

    # Initializing the figure
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Blues",
            hovertemplate="Pred: %{x}<br>True: %{y}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted label",
        yaxis_title="True label",
        autosize=True,
    )
    return fig


# Function to evaluate a single model
def evaluate_single_model(model: Any, X_test, y_test, model_name: str | None = None):

    # Measuring prediction time
    start_time = time.time()

    # Starting predictions
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time

    # Scores / probabilities for ROC-AUC
    y_scores = get_probability(model, X_test)

    # Standard classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # Weighted F1 Score (For multi class classification)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (For binary or multi-class if possible)
    roc_auc = np.nan

    # Checcking if we even have score/probabilities
    if y_scores is not None:
        try:
            # Case 1
            if y_scores.ndim == 1:
                roc_auc = float(roc_auc_score(y_test, y_scores))
            else:
                # Case 2: probability matrix
                if y_scores.shape[1] == 2:
                    # Binary: use positive class probability
                    roc_auc = float(roc_auc_score(y_test, y_scores[:, 1]))
                else:
                    # Multi-class: one-vs-rest, weighted average
                    roc_auc = float(
                        roc_auc_score(
                            y_test,
                            y_scores,
                            multi_class="ovr",
                            average="weighted",
                        )
                    )
        except Exception:
            roc_auc = np.nan

    # Confusion matrix and figure
    cm = confusion_matrix(y_test, y_pred)
    labels = getattr(model, "classes_", None)
    cm_fig = plot_confusion_matrix(
        cm,
        labels=labels,
        title=f"Confusion Matrix - {model_name or type(model).__name__}",
    )

    # Collect all metrics into one dict
    metrics = {
        "model": model_name or type(model).__name__,
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else np.nan,
        "inference_time": float(inference_time),
    }

    # Add training time if it was stored during training
    training_time = getattr(model, "_training_time", None)
    if training_time is not None:
        metrics["training_time"] = float(training_time)

    return metrics, cm_fig


def evaluate_all_models(models: dict, X_test, y_test):
    """
    Evaluate multiple models on the same test set.

    models: {model_name: fitted_model}

    Returns:
    - results_df: one row per model with metrics
    - confusion_figs: {model_name: Plotly figure}
    """
    # List to store all records
    records = []
    # Disttionary to store model name and its confusion matrix figur
    confusion_figs = {}

    # Making predictions on each model and evaluating them
    for name, model in models.items():
        metrics, cm_fig = evaluate_single_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
        )
        records.append(metrics)
        confusion_figs[name] = cm_fig

    results_df = pd.DataFrame.from_records(records)
    return results_df, confusion_figs