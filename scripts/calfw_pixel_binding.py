"""Exact decoded-pixel joins to named CALFW sources, not face recognition.

These helpers do not authenticate source downloads. Source provenance must be
verified separately before a mapping may support benchmark subject inference.
No resizing, channel swapping or approximate matching occurs here.
"""
from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from scripts.calfw_protocol import person_of


def pixel_key(pixels: np.ndarray) -> str:
    pixels = np.asarray(pixels)
    if (pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3
            or not pixels.shape[0] or not pixels.shape[1]):
        raise ValueError('nonempty decoded uint8 HxWx3 image required')
    digest = hashlib.sha256(struct.pack('<III', *pixels.shape))
    digest.update(np.ascontiguousarray(pixels).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class NamedPixelIndex:
    persons_by_pixel: dict[str, frozenset[str]]
    names_by_pixel: dict[str, frozenset[str]]
    source_records: int


def build_index(sources: Iterable[tuple[str, np.ndarray]]) -> NamedPixelIndex:
    persons: dict[str, set[str]] = defaultdict(set)
    names: dict[str, set[str]] = defaultdict(set)
    seen = {}
    records = 0
    for name, pixels in sources:
        person = person_of(name)
        key = pixel_key(pixels)
        if name in seen and seen[name] != key:
            raise ValueError('same source image name has conflicting decoded pixels')
        seen[name] = key
        persons[key].add(person)
        names[key].add(name)
        records += 1
    if not records:
        raise ValueError('nonempty named source inventory required')
    return NamedPixelIndex({k: frozenset(v) for k, v in persons.items()},
                           {k: frozenset(v) for k, v in names.items()}, records)


def match_endpoints(index: NamedPixelIndex, endpoints: Iterable[np.ndarray]) -> tuple[list[str | None], dict]:
    """Return private person tokens and aggregate coverage; never resolve ambiguity."""
    mapping = []
    missing = ambiguous = image_aliases = 0
    for pixels in endpoints:
        key = pixel_key(pixels)
        candidates = index.persons_by_pixel.get(key, frozenset())
        if not candidates:
            missing += 1
            mapping.append(None)
        elif len(candidates) != 1:
            ambiguous += 1
            mapping.append(None)
        else:
            mapping.append(next(iter(candidates)))
            image_aliases += int(len(index.names_by_pixel[key]) > 1)
    if not mapping:
        raise ValueError('nonempty endpoint inventory required')
    return mapping, {
        'endpoints': len(mapping), 'matched_person_endpoints': len(mapping) - missing - ambiguous,
        'unmatched_endpoints': missing, 'ambiguous_person_endpoints': ambiguous,
        'same_person_image_alias_endpoints': image_aliases,
        'named_source_records': index.source_records,
        'exact_pixel_person_coverage_complete': not missing and not ambiguous,
        'source_provenance_verified_by_this_helper': False,
        'subject_ci_available': False, 'publication_ready': False,
    }


def check_pair_labels(mapping: list[str | None], labels: np.ndarray) -> dict:
    """Check only resolved pairs; missing metadata never validates a pair."""
    labels = np.asarray(labels)
    if (labels.ndim != 1 or labels.dtype.kind not in 'biu'
            or not np.isin(labels, [0, 1]).all() or len(mapping) != 2 * len(labels)
            or not len(labels)):
        raise ValueError('binary labels and exactly two mapped endpoints per pair required')
    if any(p is not None and (not isinstance(p, str) or not p) for p in mapping):
        raise ValueError('person tokens must be nonempty strings or None')
    resolved = missing = conflicts = 0
    for i, label in enumerate(labels):
        a, b = mapping[2 * i:2 * i + 2]
        if a is None or b is None:
            missing += 1
        else:
            resolved += 1
            conflicts += int((a == b) != bool(label))
    return {'pairs': len(labels), 'resolved_pairs_checked': resolved,
            'unresolved_pairs': missing, 'identity_label_conflicts': conflicts,
            'all_pair_labels_verified': not missing and not conflicts}
