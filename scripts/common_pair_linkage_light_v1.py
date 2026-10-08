"""Pure-array/JSON linkage checks: no Torch, neural inference or training imports.

Exact validation semantics ported from run_common_mechanism_v1.validate_pairs and
evaluate_oriented_cuda_v1.validate_written_inputs. Existing bound sources stay intact.
"""

import json

import numpy as np


def validate_pairs(left, right, labels, a, b, people):
    left, right, labels = np.asarray(left), np.asarray(right), np.asarray(labels)
    if (left.ndim != 1 or right.shape != left.shape or labels.shape != left.shape
            or left.dtype.kind not in "iu" or right.dtype.kind not in "iu"
            or labels.dtype.kind not in "iu" or set(np.unique(labels)) != {0, 1}
            or np.any(left < 0) or np.any(right < 0)
            or np.any(left >= len(people)) or np.any(right >= len(people))):
        raise ValueError("valid shared image-index pairs required")
    if not np.array_equal(a, people[left]) or not np.array_equal(b, people[right]):
        raise ValueError("pair endpoint metadata not linked to image indices")
    if not np.array_equal(labels, (people[left] == people[right]).astype(int)):
        raise ValueError("pair labels contradict source people")


def validate_written_inputs(target, expected):
    actual = json.loads(target.read_text(encoding="utf-8"))["inputs"]
    if actual != expected:
        target.unlink()  # Only the caller's fresh completion marker.
        raise RuntimeError("inputs changed during manifest publication")
