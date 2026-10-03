"""Audit official CALFW pair metadata, without asserting a local .bin binding.

The official text lists positives first, with positive-fold tokens 1..10;
all negatives have token 0. Negative fold membership is NOT encoded here.
Person tokens returned by parse_pairs are private metadata, not public output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from age_gap.common.manifest import file_record, write_experiment_manifest

OFFICIAL_PAIRS_SHA256 = "871a2f047d57754bd8ed176b9bbcdb42a7b9ba03cd839702c0a28dfdebc9f8fc"
SOURCE_URL = "https://drive.google.com/file/d/1_cYgy7VFCy6JqkR8EvOxCVS02jHN1ozm/view"
ZIP_MEMBER = "calfw/pairs_CALFW.txt"
NAME = re.compile(r"(?P<person>[A-Za-z0-9][A-Za-z0-9_-]*)_(?P<index>\d{4})\.jpg")


def person_of(name: str) -> str:
    match = NAME.fullmatch(name)
    if match is None or int(match['index']) < 1:
        raise ValueError("invalid CALFW image filename")
    return match['person']


@dataclass(frozen=True)
class ProtocolPair:
    image_a: str
    image_b: str
    source_token: int

    @property
    def person_a(self) -> str:
        return person_of(self.image_a)

    @property
    def person_b(self) -> str:
        return person_of(self.image_b)

    @property
    def is_same(self) -> bool:
        return self.source_token != 0

    @property
    def positive_fold(self) -> int | None:
        return self.source_token if self.is_same else None


def parse_pairs(raw: bytes) -> list[ProtocolPair]:
    lines = raw.decode('ascii').splitlines()
    if not lines or len(lines) % 2:
        raise ValueError("nonempty even endpoint line count required")
    parsed = []
    for offset in range(0, len(lines), 2):
        endpoints = [line.split() for line in lines[offset:offset + 2]]
        if any(len(row) != 2 for row in endpoints):
            raise ValueError("expected filename and source token on each endpoint line")
        a, b = endpoints
        if not re.fullmatch(r'(?:[0-9]|10)', a[1]) or a[1] != b[1]:
            raise ValueError("matching source tokens in 0..10 required")
        pair = ProtocolPair(a[0], b[0], int(a[1]))
        same_person = pair.person_a == pair.person_b
        if same_person != pair.is_same or pair.image_a == pair.image_b:
            raise ValueError("identity/self-image contradiction in official pair metadata")
        parsed.append(pair)
    return parsed


def summarize(pairs: list[ProtocolPair]) -> dict:
    if len(pairs) != 6000:
        raise ValueError("official release requires 6000 pairs")
    expected = [fold for fold in range(1, 11) for _ in range(300)] + [0] * 3000
    if [p.source_token for p in pairs] != expected:
        raise ValueError("unexpected official positive-block/negative-block ordering")
    names = Counter(name for pair in pairs for name in (pair.image_a, pair.image_b))
    return {
        'pairs': len(pairs), 'positive_pairs': 3000, 'negative_pairs': 3000,
        'unique_pair_images': len(names),
        'unique_pair_person_tokens': len({person_of(name) for name in names}),
        'image_multiplicity_histogram': {
            str(k): v for k, v in sorted(Counter(names.values()).items())
        },
        'positive_fold_counts': {str(k): 300 for k in range(1, 11)},
        'negative_fold_assignment': 'not encoded in source text',
        'scope': 'official pair text structure; not a local cache endpoint mapping',
        'bin_binding_verified': False, 'subject_ci_available': False,
        'publication_ready': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('fresh output directory required')
    inputs = [args.pairs, Path(__file__)]
    before = [file_record(p) for p in inputs]
    raw = args.pairs.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OFFICIAL_PAIRS_SHA256:
        raise ValueError('official acquired pair-file checksum mismatch')
    result = summarize(parse_pairs(raw))
    if before != [file_record(p) for p in inputs]:
        raise ValueError('inputs changed during audit')
    args.out.mkdir(parents=True)
    target = args.out / 'summary.json'
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    manifest = write_experiment_manifest(
        args.out / 'summary.manifest.json', experiment='calfw-official-pair-text-audit',
        parameters={'source_url': SOURCE_URL, 'zip_member': ZIP_MEMBER,
                    'expected_pair_file_sha256': OFFICIAL_PAIRS_SHA256,
                    'endpoint_binding_attempted': False},
        metrics=result, inputs=inputs, outputs=[target],
    )
    if (before != [file_record(p) for p in inputs]
            or json.loads(manifest.read_text(encoding='utf-8'))['inputs'] != before):
        manifest.unlink()
        raise ValueError('inputs changed before completed manifest')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
