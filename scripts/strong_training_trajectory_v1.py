"""Describe completed native training histories without inventing validation loss.

This reducer does not verify source manifests or establish a failure mechanism.
Callers must independently verify the native training artifact before using output.
"""

import math


def summarize(history, *, epochs=8):
    if len(history) != epochs or [r.get("epoch") for r in history] != list(range(1, epochs + 1)):
        raise ValueError("complete ordered fixed-budget history required")
    required = ("training_loss", "validation_auc", "mean_gradient_norm", "epoch_seconds")
    for row in history:
        for key in required:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("finite numeric training measurements required")
        if not 0 <= row["validation_auc"] <= 1 or any(
            row[k] < 0 for k in ("training_loss", "mean_gradient_norm", "epoch_seconds")
        ):
            raise ValueError("measurements outside valid ranges")
    first, last = history[0], history[-1]
    best = max(history, key=lambda r: r["validation_auc"])
    return {
        "version": "strong-training-trajectory-v1",
        "epochs": epochs,
        "selected_epoch": epochs,
        "checkpoint_selection": "last_epoch",
        "training_loss_first": first["training_loss"],
        "training_loss_last": last["training_loss"],
        "training_loss_change": last["training_loss"] - first["training_loss"],
        "validation_auc_first": first["validation_auc"],
        "validation_auc_last": last["validation_auc"],
        "validation_auc_change": last["validation_auc"] - first["validation_auc"],
        "diagnostic_peak_auc_epoch": best["epoch"],
        "diagnostic_peak_auc": best["validation_auc"],
        "last_minus_peak_auc": last["validation_auc"] - best["validation_auc"],
        "validation_loss": None,
        "validation_loss_status": "not collected by v1 trainer; cannot infer from AUC",
        "history": [{k: row[k] for k in ("epoch", *required)} for row in history],
        "limitation": "descriptive recorded history; peak is not checkpoint selection; no mechanism conclusion",
    }
