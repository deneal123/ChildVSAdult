"""Bounded singleton-profile repair for new exact-age controls, never active arms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from age_gap.common.io import PROJECT_ROOT, read_jsonl, write_jsonl
from age_gap.common.manifest import file_record, write_experiment_manifest
from scripts.audit_exact_negative_matching import exact_arm
from scripts.build_matched_agegap_arms import _endpoint_pool, _sample_impostors


def valid_positive(row, groups):
    return not (
        row.get("split") != "train"
        or type(row.get("label")) is not int
        or row["label"] != 1
        or row.get("status", "ok") != "ok"
        or any(type(row.get(f"age_{s}")) is not int or row[f"age_{s}"] < 0 for s in ("a", "b"))
        or any(
            not isinstance(row.get(f"face_{s}"), str) or not row[f"face_{s}"] for s in ("a", "b")
        )
        or row["face_a"] == row["face_b"]
        or any(row.get(f"identity_group_{s}") not in groups for s in ("a", "b"))
        or groups[row["identity_group_a"]] != groups[row["identity_group_b"]]
        or row.get("age_gap") != abs(row["age_a"] - row["age_b"])
    )


def profile(rows, groups):
    return {
        "positive_pairs": len(rows),
        "images": len({r[f"face_{s}"] for r in rows for s in ("a", "b")}),
        "recorded_people": len({groups[r["identity_group_a"]] for r in rows}),
    }


def unique_quality(rows):
    """Exclude duplicate face keys entirely; never choose first/last observations."""
    unique, duplicate_faces = {}, set()
    duplicate_rows = conflicting_rows = 0
    for row in rows:
        face = row["face_id"]
        if face in unique:
            duplicate_rows += 1
            conflicting_rows += int(unique[face] != row)
            duplicate_faces.add(face)
        else:
            unique[face] = row
    for face in duplicate_faces:
        del unique[face]
    return unique, {
        "duplicate_rows": duplicate_rows,
        "excluded_duplicate_faces": len(duplicate_faces),
        "conflicting_rows": conflicting_rows,
        "policy": "all duplicate face keys excluded from replacement eligibility",
    }


def repair(arms, groups, canonical, quality, unsupported, *, seed=42, max_attempts=64):
    if set(arms) != {"LOW", "CROSS"} or set(unsupported) != set(arms):
        raise ValueError("LOW/CROSS arms and unsupported sets required")
    if type(seed) is not int or type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("integer seed and positive integer attempt limit required")
    if any(not isinstance(person, str) or not person for person in groups.values()):
        raise ValueError("nonempty recorded-person mapping required")
    heldout = [[r for r in arms[a] if r["split"] != "train"] for a in ("LOW", "CROSS")]
    if heldout[0] != heldout[1]:
        raise ValueError("shared heldout required")
    positive = {a: [r for r in arms[a] if r["split"] == "train" and r["label"] == 1] for a in arms}
    for arm, rows in arms.items():
        if len({r["pair_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate source pair ID")
        if not positive[arm] or not all(valid_positive(r, groups) for r in positive[arm]):
            raise ValueError("invalid source positives")
    if profile(positive["LOW"], groups) != profile(positive["CROSS"], groups):
        raise ValueError("original arm profiles unequal")
    selected_people = {groups[r["identity_group_a"]] for rows in positive.values() for r in rows}
    heldout_people = {groups[r[f"identity_group_{s}"]] for r in heldout[0] for s in ("a", "b")}
    heldout_faces = {r[f"face_{s}"] for r in heldout[0] for s in ("a", "b")}
    selected_faces = {
        r[f"face_{s}"] for rows in positive.values() for r in rows for s in ("a", "b")
    }
    if selected_people & heldout_people or selected_faces & heldout_faces:
        raise ValueError("source train/heldout recorded-person or image overlap")
    if {groups[r["identity_group_a"]] for r in positive["LOW"]} & {
        groups[r["identity_group_a"]] for r in positive["CROSS"]
    }:
        raise ValueError("source arm recorded-person overlap")
    candidates = [r for r in canonical if valid_positive(r, groups)]
    if len({r["pair_id"] for r in candidates}) != len(candidates):
        raise ValueError("duplicate canonical positive pair ID")
    observations, regimes = defaultdict(set), defaultdict(set)
    for row in candidates:
        person = groups[row["identity_group_a"]]
        if 1 <= row["age_gap"] <= 2:
            regimes[person].add("LOW")
        if row["age_gap"] >= 25:
            regimes[person].add("CROSS")
        for side in ("a", "b"):
            observations[row[f"face_{side}"]].add((row[f"age_{side}"], person))
    ambiguous = {face for face, values in observations.items() if len(values) != 1}
    excluded_faces = ambiguous | selected_faces | heldout_faces
    excluded_ids = {r["pair_id"] for rows in arms.values() for r in rows}
    forbidden = {
        tuple(sorted((r["face_a"], r["face_b"]))) for r in canonical if r.get("label") == 1
    }
    result, output, replacements = {}, {}, []
    used_people = set(selected_people)
    for arm in ("LOW", "CROSS"):
        original = positive[arm]
        ids = {r["pair_id"] for r in original}
        if not set(unsupported[arm]) <= ids:
            raise ValueError("unsupported target not in source positives")
        affected = {
            groups[r["identity_group_a"]] for r in original if r["pair_id"] in unsupported[arm]
        }
        by_person = defaultdict(list)
        for row in original:
            by_person[groups[row["identity_group_a"]]].append(row)
        if any(len(by_person[person]) != 1 for person in affected):
            raise ValueError("bounded singleton repair cannot silently alter richer profiles")
        retained = [r for r in original if groups[r["identity_group_a"]] not in affected]
        ages = {r[f"age_{s}"] for r in retained for s in ("a", "b")}
        eligible = []
        excluded_people = used_people | heldout_people
        for row in candidates:
            person = groups[row["identity_group_a"]]
            if (
                person in excluded_people
                or row["pair_id"] in excluded_ids
                or regimes[person] != {arm}
                or not (1 <= row["age_gap"] <= 2 if arm == "LOW" else row["age_gap"] >= 25)
                or row["age_b"] not in ages
                or any(row[f"face_{s}"] in excluded_faces for s in ("a", "b"))
                or any(
                    quality.get(row[f"face_{s}"], {}).get("is_usable") is not True
                    for s in ("a", "b")
                )
            ):
                continue
            eligible.append(row)
        eligible.sort(
            key=lambda row: hashlib.sha256(f"{seed}:{arm}:{row['pair_id']}".encode()).hexdigest()
        )
        unique = {}
        for row in eligible:
            unique.setdefault(groups[row["identity_group_a"]], row)
        eligible = list(unique.values())
        old_slots = [
            i for i, row in enumerate(original) if groups[row["identity_group_a"]] in affected
        ]
        success = None
        for attempt in range(min(max_attempts, max(0, len(eligible) - len(old_slots) + 1))):
            proposal = copy.deepcopy(original)
            for i, slot in enumerate(old_slots):
                proposal[slot] = copy.deepcopy(eligible[attempt + i])
            enriched = [{**r, "_person_id": groups[r["identity_group_a"]]} for r in proposal]
            pool, conflicts = _endpoint_pool(enriched, groups)
            if conflicts or profile(proposal, groups) != profile(original, groups):
                raise ValueError("replacement violated age consistency or exact profile")
            try:
                negatives, diagnostic = _sample_impostors(
                    enriched, pool, arm=arm, seed=seed, bin_years=10, max_age_error=0
                )
            except ValueError:
                continue
            if any(tuple(sorted((r["face_a"], r["face_b"]))) in forbidden for r in negatives):
                continue
            for row in negatives:
                row["pair_id"] = "repaired_" + row["pair_id"]
                row["pair_type"] = "negative_exact_age_repaired_pool_impostor"
            prepared = proposal + negatives + copy.deepcopy(heldout[0])
            if len({r["pair_id"] for r in prepared}) != len(prepared):
                raise ValueError("repaired pair ID collision")
            exact_report, _, _ = exact_arm(prepared, groups, arm=arm, forbidden=forbidden)
            if not exact_report["full_fixed_positive_protocol_feasible"]:
                continue
            success = (prepared, proposal, diagnostic, exact_report, attempt + 1)
            break
        result[arm] = {
            "affected_singleton_people": len(affected),
            "eligible_new_people": len(eligible),
            "maximum_candidate_attempts": max_attempts,
            "complete": success is not None,
            "source_profile": profile(original, groups),
        }
        if success is None:
            return (
                {
                    "arms": result,
                    "complete": False,
                    "training_ready": False,
                    "publication_ready": False,
                    "scope": "bounded repair failed; not global positive-selection impossibility",
                },
                None,
                [],
            )
        prepared, proposal, diagnostic, exact_report, attempts = success
        result[arm].update(
            repaired_profile=profile(proposal, groups),
            attempts=attempts,
            negative_diagnostics=diagnostic,
            global_exact_matching=exact_report,
        )
        output[arm] = prepared
        used_people.update(groups[r["identity_group_a"]] for r in proposal)
        for slot in old_slots:
            replacements.append(
                {
                    "arm": arm,
                    "old_positive_pair_id": original[slot]["pair_id"],
                    "new_positive_pair_id": proposal[slot]["pair_id"],
                }
            )
    return (
        {
            "arms": result,
            "complete": True,
            "training_ready": False,
            "publication_ready": False,
            "replacement_selection": f"seed{seed} hash order; singleton pair/two-image profiles; no recognition outcomes",
            "heldout_unchanged": True,
            "known_genuine_negative_edges_excluded": True,
            "identity_purity": "recorded-person mapping only; human/benchmark audits pending",
            "scope": "fresh exact-supported control preparation, not training or quality-balance proof",
        },
        output,
        replacements,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-attempts", type=int, default=64)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh destination required")
    root = PROJECT_ROOT
    prerequisite = root / "metrics/exact_negative_matching_20261003/summary.manifest.json"
    native = json.loads(prerequisite.read_text(encoding="utf-8"))
    if native["experiment"] != "fixed-pool-exact-negative-global-matching":
        raise ValueError("wrong prerequisite")
    records = native["inputs"] + native["outputs"]
    paths = [
        Path(r["path"]) if Path(r["path"]).is_absolute() else root / r["path"] for r in records
    ]
    if [file_record(path) for path in paths] != records:
        raise ValueError("prerequisite records changed")
    consumed = [
        root / "data/processed/person_clusters.jsonl",
        root / "data/processed/pairs.jsonl",
        root / "data/interim/face_quality_audit.jsonl",
    ]
    for arm in ("low", "cross"):
        consumed.extend(
            [
                root / f"data/interim/restricted_matched_arms_20261003/private/{arm}_arm.jsonl",
                root / f"metrics/exact_negative_matching_20261003/private/{arm}_assignment.jsonl",
            ]
        )
    if not {path.resolve() for path in consumed} <= {path.resolve() for path in paths}:
        raise ValueError("consumed source absent from verified prerequisite records")
    inputs = [*paths, prerequisite, Path(__file__), root / "scripts/build_matched_agegap_arms.py"]
    inputs = list(dict.fromkeys(path.resolve() for path in inputs))
    before = [file_record(path) for path in inputs]
    groups = {}
    for row in read_jsonl(root / "data/processed/person_clusters.jsonl"):
        key, person = row["identity_group_id"], row["person_id"]
        if key in groups and groups[key] != person:
            raise ValueError("conflicting recorded-person mapping")
        groups[key] = person
    arms = {
        a: list(
            read_jsonl(
                root
                / f"data/interim/restricted_matched_arms_20261003/private/{a.lower()}_arm.jsonl"
            )
        )
        for a in ("LOW", "CROSS")
    }
    unsupported = {
        a: [
            r["target_positive_pair_id"]
            for r in read_jsonl(
                root
                / f"metrics/exact_negative_matching_20261003/private/{a.lower()}_assignment.jsonl"
            )
            if not r["matched"]
        ]
        for a in arms
    }
    quality, quality_audit = unique_quality(
        read_jsonl(root / "data/interim/face_quality_audit.jsonl")
    )
    canonical = list(read_jsonl(root / "data/processed/pairs.jsonl"))
    result, prepared, replacements = repair(
        arms, groups, canonical, quality, unsupported, max_attempts=args.max_attempts
    )
    result["replacement_quality_sidecar"] = quality_audit
    if before != [file_record(path) for path in inputs]:
        raise ValueError("repair inputs changed")
    private = args.out / "private"
    private.mkdir(parents=True)
    outputs = []
    if prepared is not None:
        for arm in prepared:
            path = private / f"{arm.lower()}_candidate_arm.jsonl"
            write_jsonl(path, prepared[arm])
            outputs.append(path)
    replacement_path = private / "replacements.jsonl"
    write_jsonl(replacement_path, replacements)
    output = args.out / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    outputs.extend([replacement_path, output])
    target = args.out / "summary.manifest.json"
    write_experiment_manifest(
        target,
        experiment="exact-supported-singleton-profile-repair-v2-row-regime",
        parameters={
            "seed": 42,
            "max_attempts_per_arm": args.max_attempts,
            "training_executed": False,
        },
        metrics=result,
        inputs=inputs,
        outputs=outputs,
    )
    if before != json.loads(target.read_text(encoding="utf-8"))["inputs"]:
        target.unlink()
        raise ValueError("inputs changed during manifest write; marker withdrawn")
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "arms": {
                    arm: {
                        k: row[k]
                        for k in (
                            "affected_singleton_people",
                            "eligible_new_people",
                            "complete",
                            "source_profile",
                        )
                    }
                    for arm, row in result["arms"].items()
                },
                "training_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
