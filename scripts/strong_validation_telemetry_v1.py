"""Future same-pass validation telemetry; no native producer integration yet.

The pinned trainer's validation iterator already consumes one CPU RNG base-seed
draw. Replacing that one pass preserves its loader/forward ordering; a second
pass would not. Existing component histories must remain unchanged. A future
versioned producer must bind this helper and separately publish these records.
"""

import math
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader

from age_gap.evaluation.metrics import roc_auc

VERSION = "strong-same-validation-pass-telemetry-v1"


class ValidationCollector:
    """Collect global weighted loss from the same cosine values used for AUC.

    Finite two-class validation is required, unlike the legacy empty/single-class
    NaN allowance. This is a new versioned precondition, not universal parity.
    Nonuniform-weight global loss is intentionally distinct from the legacy
    trainer's pair-count-weighted average of batch weighted means.
    """

    def __init__(self, margin=0.3):
        if type(margin) is not float or not math.isfinite(margin) or margin != 0.3:
            raise ValueError("fixed margin0.3 required")
        self.margin = margin
        self.records = []

    def __call__(self, backbone, ds, device, batch_size):
        if not len(ds) or type(batch_size) is not int or batch_size < 1:
            raise ValueError("nonempty validation and positive integer batch required")
        if set(ds.labels) != {0, 1}:
            raise ValueError("two-class validation required by telemetry contract")
        backbone.eval()
        # Intentionally identical defaults/order to pinned finetune._val_auc.
        # Do not add generator= (that would remove its existing global draw).
        loader = DataLoader(ds, batch_size=batch_size)
        scores, labels = [], []
        numerator, denominator, count = 0.0, 0.0, 0
        with torch.no_grad():
            for ta, tb, y, w in loader:
                za, zb = backbone(ta.to(device)), backbone(tb.to(device))
                cosine = (za * zb).sum(dim=-1)
                y, w = y.to(device), w.to(device)
                if (
                    cosine.ndim != 1
                    or y.shape != cosine.shape
                    or w.shape != cosine.shape
                    or not torch.isfinite(cosine).all()
                    or not torch.isfinite(w).all()
                    or not ((y == 0) | (y == 1)).all()
                    or not (w > 0).all()
                ):
                    raise ValueError("finite cosines/binary labels/positive weights required")
                per_pair = y * (1 - cosine) + (1 - y) * torch.relu(cosine - self.margin)
                numerator += float((per_pair * w).sum())
                denominator += float(w.sum())
                count += len(y)
                scores.extend(cosine.cpu().tolist())
                labels.extend(y.cpu().tolist())
        if count != len(ds) or not np.array_equal(np.asarray(labels), np.asarray(ds.labels)):
            raise ValueError("validation count/label order differs from dataset metadata")
        auc = roc_auc(np.asarray(scores), np.asarray(ds.labels))
        loss = numerator / denominator
        if not all(math.isfinite(x) for x in (auc, loss, numerator, denominator)):
            raise ValueError("finite global loss and AUC required")
        self.records.append(
            {
                "epoch": len(self.records) + 1,
                "validation_loss": loss,
                "validation_numerator": numerator,
                "validation_denominator": denominator,
                "validation_pairs": count,
                "validation_auc": auc,
            }
        )
        return auc


@contextmanager
def installed_validation_collector(trainer, collector):
    """Process-local replacement; restore the original seam on every exit.

    Not thread-safe: only a future dedicated producer process may use this.
    Never use it to reinterpret or mutate an already completed native run.
    """
    if not isinstance(collector, ValidationCollector) or collector.records:
        raise ValueError("fresh validation collector required")
    original = trainer._val_auc
    trainer._val_auc = collector
    try:
        yield collector
    finally:
        trainer._val_auc = original
